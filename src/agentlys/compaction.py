"""Client-side conversation compaction for agentlys."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentlys.base import AgentlysBase

logger = logging.getLogger(__name__)


@runtime_checkable
class CompactionHandler(Protocol):
    """Protocol for custom compaction implementations.

    Implement both methods to create a custom compaction strategy.
    """

    async def should_compact(self, chat: AgentlysBase) -> bool:
        """Return True if the conversation should be compacted."""
        ...

    async def compact(self, chat: AgentlysBase) -> None:
        """Compact the conversation in place by mutating chat.messages."""
        ...


DEFAULT_COMPACTION_PROMPT = (
    "You are summarizing a conversation to preserve continuity. "
    "The summary will replace the original messages, so include: "
    "current state, decisions made, key findings, next steps, "
    "and any code snippets or variable names that are still relevant. "
    "Be concise but thorough. Wrap your summary in <summary></summary> tags."
)


@dataclass
class TokenThresholdCompaction:
    """Client-side compaction using a cheap model for summarization.

    Checks the most recent API response's ``usage.input_tokens`` against a
    configurable threshold. When exceeded, summarizes the entire conversation
    into a single compaction message.

    Works with any provider implementing ``BaseProvider.complete()``
    (Anthropic, OpenAI, and any OpenAI-compatible API).

    After a compaction, preserved attachments (e.g. PDF document parts) keep
    contributing a fixed token floor to every request that no amount of
    summarization can reduce. To avoid re-compacting on every turn once that
    floor exceeds the threshold, ``should_compact`` measures the floor as the
    first response's usage after the compaction message and triggers only when
    the conversation has grown ``token_threshold`` tokens beyond it.

    Args:
        token_threshold: Trigger compaction when input tokens exceed this value
            (after a compaction: when input tokens grow this much beyond the
            post-compaction floor).
        summary_model: Model to use for generating summaries (cheap/fast
            recommended). Defaults to the provider's current model — pass a
            cheap model explicitly to reduce summarization cost.
        instructions: Custom summarization prompt. Replaces the default if provided.
    """

    token_threshold: int = 100_000
    summary_model: Optional[str] = None
    instructions: Optional[str] = None

    @staticmethod
    def _total_input_tokens(usage: dict) -> int:
        """Total input tokens including cached tokens (cache_creation_input_tokens,
        cache_read_input_tokens), which are separate from input_tokens when
        prompt caching is enabled."""
        return (
            usage["input_tokens"]
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )

    async def should_compact(self, chat: AgentlysBase) -> bool:
        """Check if compaction is needed based on the last API response's token usage.

        Each assistant Message from the provider carries a ``usage`` dict.
        """
        latest = None
        for msg in reversed(chat.messages):
            if msg.usage and "input_tokens" in msg.usage:
                latest = self._total_input_tokens(msg.usage)
                break
        if latest is None:
            return False

        # Post-compaction floor: preserved document parts (and the summary
        # itself) are re-sent with every request, so their tokens show up in
        # usage without being reducible by another compaction. Subtract the
        # first post-compaction response's usage so only real conversation
        # growth counts toward the threshold.
        baseline = 0
        if chat.messages[0].has_compaction:
            for msg in chat.messages:
                if msg.usage and "input_tokens" in msg.usage:
                    baseline = self._total_input_tokens(msg.usage)
                    break

        return latest - baseline > self.token_threshold

    async def compact(self, chat: AgentlysBase) -> None:
        """Summarize older messages and replace them with a compaction message."""
        from agentlys.model import Message, MessagePart

        messages = chat.messages
        if not messages:
            return

        # Summarize the entire conversation into a single compaction message.
        # Skip messages with no parts (e.g. after thinking block removal).
        conversation_text = []
        for msg in messages:
            if not msg.parts:
                continue
            conversation_text.append(msg.to_markdown())
        conversation_str = "\n".join(conversation_text)

        prompt = self.instructions or DEFAULT_COMPACTION_PROMPT

        summary_messages = [
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n--- Conversation to summarize ---\n{conversation_str}"
                ),
            }
        ]

        # The provider handles client shape, proxy auth and custom base_url
        summary_text = await chat.provider.complete(
            messages=summary_messages,
            # Include the original system instruction for context
            system=chat.instruction,
            model=self.summary_model,
            max_tokens=4096,
        )

        # Try to extract from <summary> tags if present
        match = re.search(r"<summary>(.*?)</summary>", summary_text, re.DOTALL)
        if match:
            summary_text = match.group(1).strip()

        # Preserve document attachments: their binary content can't be
        # captured by the text summary, so re-attach them after the summary.
        # The text history is compacted, but the model keeps direct access to
        # the original documents.
        document_parts = []
        seen_documents = set()
        for msg in messages:
            for part in msg.parts:
                if part.type != "document" or part.document is None:
                    continue
                key = (
                    part.document.name,
                    part.document.media_type,
                    part.document.data,
                )
                if key in seen_documents:
                    continue
                seen_documents.add(key)
                document_parts.append(part)

        # Build compaction message and replace conversation history
        compaction_message = Message(
            role="user",
            parts=[
                MessagePart(type="compaction", content=summary_text),
                *document_parts,
            ],
        )

        chat.messages = [compaction_message]
        logger.info(
            "Compacted %d messages into summary (%d documents preserved)",
            len(messages),
            len(document_parts),
        )
