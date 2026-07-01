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

    Args:
        token_threshold: Trigger compaction when input tokens exceed this value.
        summary_model: Model to use for generating summaries (cheap/fast
            recommended). Defaults to the provider's current model — pass a
            cheap model explicitly to reduce summarization cost.
        instructions: Custom summarization prompt. Replaces the default if provided.
    """

    token_threshold: int = 100_000
    summary_model: Optional[str] = None
    instructions: Optional[str] = None

    async def should_compact(self, chat: AgentlysBase) -> bool:
        """Check if compaction is needed based on the last API response's token usage.

        Each assistant Message from the provider carries a ``usage`` dict.
        Total input tokens includes cached tokens (cache_creation_input_tokens,
        cache_read_input_tokens) which are separate from input_tokens when
        prompt caching is enabled.
        """
        for msg in reversed(chat.messages):
            if msg.usage and "input_tokens" in msg.usage:
                total = (
                    msg.usage["input_tokens"]
                    + msg.usage.get("cache_creation_input_tokens", 0)
                    + msg.usage.get("cache_read_input_tokens", 0)
                )
                return total > self.token_threshold
        return False

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

        # Build compaction message and replace conversation history
        compaction_message = Message(
            role="user",
            parts=[MessagePart(type="compaction", content=summary_text)],
        )

        chat.messages = [compaction_message]
        logger.info(
            "Compacted %d messages into summary",
            len(messages),
        )
