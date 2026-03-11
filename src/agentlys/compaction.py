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

    Counts tokens via Anthropic's /v1/messages/count_tokens endpoint.
    When threshold is exceeded, summarizes older messages using summary_model
    and replaces them with a compaction message, preserving the last N messages.

    Args:
        token_threshold: Trigger compaction when input tokens exceed this value.
        summary_model: Model to use for generating summaries (cheap/fast recommended).
        preserve_last_n: Number of recent messages to keep verbatim alongside the summary.
        instructions: Custom summarization prompt. Replaces the default if provided.
    """

    token_threshold: int = 100_000
    summary_model: str = "claude-haiku-4-5-20251001"
    preserve_last_n: int = 4
    instructions: Optional[str] = None

    async def should_compact(self, chat: AgentlysBase) -> bool:
        """Check if compaction is needed by counting input tokens."""
        import anthropic as anthropic_sdk

        from agentlys.providers.anthropic import message_to_anthropic_dict
        from agentlys.providers.utils import add_empty_function_result

        provider = chat.provider
        client = anthropic_sdk.AsyncAnthropic(
            base_url=provider.client._base_url
            if hasattr(provider, "client")
            else None,
        )

        messages = provider.prepare_messages(
            transform_function=lambda m: message_to_anthropic_dict(m),
            transform_list_function=add_empty_function_result,
        )

        # Build system messages (same logic as _prepare_request_params)
        system_messages = []
        if chat.instruction:
            system_messages.append({"type": "text", "text": chat.instruction})
        if chat.initial_tools_states:
            system_messages.append(
                {"type": "text", "text": chat.initial_tools_states}
            )

        tools = [
            {
                "name": s["name"],
                "description": s["description"] or "No description provided",
                "input_schema": s["parameters"],
            }
            for s in chat.functions_schema
        ]

        kwargs = {}
        if system_messages:
            kwargs["system"] = system_messages
        if tools:
            kwargs["tools"] = tools

        try:
            result = await client.messages.count_tokens(
                model=provider.model,
                messages=messages,
                **kwargs,
            )
            return result.input_tokens > self.token_threshold
        except Exception:
            # Fallback: use message count heuristic if token counting fails
            logger.warning(
                "Token counting failed, falling back to message count heuristic"
            )
            return len(chat.messages) > 40

    async def compact(self, chat: AgentlysBase) -> None:
        """Summarize older messages and replace them with a compaction message."""
        import anthropic as anthropic_sdk

        from agentlys.model import Message, MessagePart

        messages = chat.messages
        if len(messages) <= self.preserve_last_n:
            return  # Nothing to compact

        # Split: messages to summarize vs messages to preserve
        to_summarize = messages[: -self.preserve_last_n]
        to_preserve = messages[-self.preserve_last_n :]

        # Build a text representation of messages to summarize
        conversation_text = []
        for msg in to_summarize:
            conversation_text.append(msg.to_markdown())
        conversation_str = "\n".join(conversation_text)

        prompt = self.instructions or DEFAULT_COMPACTION_PROMPT

        provider = chat.provider
        client = anthropic_sdk.AsyncAnthropic(
            base_url=provider.client._base_url
            if hasattr(provider, "client")
            else None,
        )

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

        response = await client.messages.create(
            model=self.summary_model,
            messages=summary_messages,
            max_tokens=4096,
            **kwargs,
        )

        # Extract summary text
        summary_text = response.content[0].text

        # Try to extract from <summary> tags if present
        match = re.search(r"<summary>(.*?)</summary>", summary_text, re.DOTALL)
        if match:
            summary_text = match.group(1).strip()

        # Build compaction message and replace conversation history
        compaction_message = Message(
            role="user",
            parts=[MessagePart(type="compaction", content=summary_text)],
        )

        chat.messages = [compaction_message] + to_preserve
        logger.info(
            "Compacted %d messages into summary, preserved last %d",
            len(to_summarize),
            len(to_preserve),
        )
