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

    Args:
        token_threshold: Trigger compaction when input tokens exceed this value.
        summary_model: Model to use for generating summaries (cheap/fast recommended).
        instructions: Custom summarization prompt. Replaces the default if provided.
    """

    token_threshold: int = 100_000
    summary_model: str = "claude-haiku-4-5-20251001"
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
        import anthropic as anthropic_sdk

        from agentlys.model import Message, MessagePart

        messages = chat.messages
        if not messages:
            return

        # Summarize the entire conversation into a single compaction message.
        conversation_text = []
        for msg in messages:
            conversation_text.append(msg.to_markdown())
        conversation_str = "\n".join(conversation_text)

        prompt = self.instructions or DEFAULT_COMPACTION_PROMPT

        provider = chat.provider
        # Reuse the provider's client so proxy auth / custom base_url work
        if hasattr(provider, "client"):
            client = provider.client
        else:
            client = anthropic_sdk.AsyncAnthropic()

        summary_messages = [
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"--- Conversation to summarize ---\n{conversation_str}"
                ),
            }
        ]

        # Include the original system instruction for context
        kwargs = {}
        if chat.instruction:
            kwargs["system"] = chat.instruction

        # If the provider exposes auth headers (e.g. proxy), inject them
        if hasattr(provider, "_get_auth_headers"):
            kwargs["extra_headers"] = await provider._get_auth_headers()

        response = await client.messages.create(
            model=self.summary_model,
            messages=summary_messages,
            max_tokens=4096,
            **kwargs,
        )

        # Extract summary text (skip ThinkingBlocks when extended thinking is enabled)
        text_block = next(
            (block for block in response.content if block.type == "text"), None
        )
        if text_block is None:
            raise RuntimeError("Compaction response contained no text block")
        summary_text = text_block.text

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
