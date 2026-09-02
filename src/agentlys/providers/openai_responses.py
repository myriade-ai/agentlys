"""OpenAI Responses API provider — GPT-5 / o-series with reasoning.

Why a second OpenAI provider: the Chat Completions API cannot round-trip a
reasoning model's hidden state between the turns of a tool loop, so every
tool result restarts the model's thinking from scratch. The Responses API
returns each turn's reasoning as an item whose ``encrypted_content`` we
replay on the next request, the same way Anthropic thinking blocks travel
with their signature.

Reasoning is carried by the existing ``thinking`` MessagePart:

- ``thinking``           → the reasoning summary text (may be empty)
- ``thinking_signature`` → ``"<reasoning item id>|<encrypted_content>"``

so callers that already persist thinking blocks for Anthropic (and reload
them with ``load_messages(keep_thinking=True)``) need no schema change.
"""

import json
import logging
import os
import typing

from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider
from agentlys.providers.openai import OPENAI_DEFAULT_BASE_URL
from agentlys.providers.utils import (
    FunctionCallParsingError,
    add_empty_function_result,
    drop_orphaned_function_results,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "16000"))

# reasoning.effort values the API accepts. Availability varies per model
# generation (``minimal`` is gpt-5 only, ``none`` needs gpt-5.1+, ``xhigh``
# gpt-5.2+); the provider forwards whatever it is given and lets the API
# reject a value the target model does not support.
_VALID_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

# Anthropic ``output_config.effort`` levels that have no OpenAI namesake.
_EFFORT_ALIASES = {"max": "xhigh"}

_SIGNATURE_SEPARATOR = "|"


def resolve_effort(value: typing.Optional[str]) -> typing.Optional[str]:
    """Normalize an effort level to what the Responses API accepts.

    Accepts the OpenAI levels as-is and translates the Anthropic-only ones,
    so a caller can keep one ``effort=`` setting across providers.
    """
    effort = value or os.getenv("AGENTLYS_EFFORT") or None
    if effort is None:
        return None
    effort = _EFFORT_ALIASES.get(effort, effort)
    if effort not in _VALID_EFFORTS:
        raise ValueError(
            f"Invalid effort {effort!r}: expected one of {_VALID_EFFORTS} "
            f"or {tuple(_EFFORT_ALIASES)}"
        )
    return effort


def thinking_to_effort(thinking: typing.Optional[dict]) -> typing.Optional[str]:
    """Map an Anthropic ``thinking`` config onto a reasoning effort.

    ``adaptive`` (let the model decide) maps to "no effort sent", which is the
    API's own adaptive default. ``enabled`` scales with the budget the caller
    had in mind. ``disabled`` becomes ``low`` rather than ``none``/``minimal``
    because neither of those exists on every reasoning model generation.
    """
    if not thinking:
        return None
    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        return "low"
    if thinking_type == "enabled":
        budget = thinking.get("budget_tokens") or 0
        if budget < 2048:
            return "low"
        if budget < 8192:
            return "medium"
        return "high"
    return None


def encode_thinking_signature(item_id: typing.Optional[str], encrypted: str) -> str:
    return f"{item_id or ''}{_SIGNATURE_SEPARATOR}{encrypted}"


def decode_thinking_signature(
    signature: typing.Optional[str],
) -> tuple[typing.Optional[str], typing.Optional[str]]:
    """Split a stored signature back into ``(item id, encrypted_content)``.

    A signature without the separator is not ours (an Anthropic one, say)
    and yields ``(None, None)`` so the block is skipped instead of replayed
    to the wrong API.
    """
    if not signature or _SIGNATURE_SEPARATOR not in signature:
        return None, None
    item_id, encrypted = signature.split(_SIGNATURE_SEPARATOR, 1)
    return (item_id or None), (encrypted or None)


def usage_to_dict(usage) -> typing.Optional[dict]:
    """Normalize Responses usage to the naming used by Message.

    ``input_tokens`` is reported as the *total* prompt size with cached
    tokens as a subset; Message uses Anthropic's split (``input_tokens`` =
    uncached remainder) so ``compaction.should_compact`` sums one shape.
    ``reasoning_tokens`` are a subset of ``output_tokens`` and are kept as an
    extra field for cost attribution.
    """
    if usage is None:
        return None

    def _get(obj, name, default=0):
        if obj is None:
            return default
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        return default if value is None else value

    input_tokens = _get(usage, "input_tokens") or 0
    output_tokens = _get(usage, "output_tokens") or 0
    cached = min(
        _get(_get(usage, "input_tokens_details", None), "cached_tokens"), input_tokens
    )
    reasoning = _get(_get(usage, "output_tokens_details", None), "reasoning_tokens")

    result = {
        "input_tokens": input_tokens - cached,
        "output_tokens": output_tokens,
    }
    if cached:
        result["cache_read_input_tokens"] = cached
    if reasoning:
        result["reasoning_tokens"] = reasoning
    return result


# ---------------------------------------------------------------------------
# Message -> Responses input items
# ---------------------------------------------------------------------------


def _content_part(part: MessagePart) -> dict:
    """Serialize a user-side content part (text, image, document)."""
    if part.type == "text":
        return {"type": "input_text", "text": part.content}
    if part.type in ("image", "function_result_image"):
        if part.image is None:
            raise ValueError(f"{part.type} part must have an image")
        return {
            "type": "input_image",
            "image_url": f"data:{part.image.format};base64,{part.image.to_base64()}",
        }
    if part.type == "document":
        if part.document is None:
            raise ValueError("Document part must have a document")
        doc = part.document
        if doc.media_type == "text/plain":
            name = doc.name or "document"
            text = doc.data.decode("utf-8", errors="replace")
            return {"type": "input_text", "text": f"[Document: {name}]\n{text}"}
        if doc.media_type == "application/pdf":
            return {
                "type": "input_file",
                "filename": doc.name or "document.pdf",
                "file_data": f"data:application/pdf;base64,{doc.to_base64()}",
            }
        raise ValueError(
            f"Unsupported document media_type {doc.media_type!r}: the Responses "
            "API accepts PDF files and inline text. Convert other formats first."
        )
    if part.type == "compaction":
        return {
            "type": "input_text",
            "text": f"[Previous conversation summary]\n{part.content}",
        }
    if part.type == "function_result":
        return {"type": "input_text", "text": part.content or ""}
    raise ValueError(f"Unknown part type: {part.type}")


def _function_result_output(parts: list[MessagePart]) -> typing.Union[str, list]:
    """Build a ``function_call_output.output`` from one call's result parts.

    Plain text stays a string (the common case, and the only form every
    OpenAI-compatible gateway accepts); an image result switches to the
    content-array form.
    """
    if all(p.type == "function_result" for p in parts):
        return "\n".join(p.content or "" for p in parts)
    output = []
    for p in parts:
        if p.type == "function_result_image":
            if p.content:
                output.append({"type": "input_text", "text": p.content})
            output.append(_content_part(p))
        else:
            output.append(_content_part(p))
    return output


def message_to_responses_items(message: Message) -> list[dict]:
    """Serialize one Message into Responses ``input`` items.

    Assistant turns fan out into several items (reasoning, function_call,
    message) because that is how the API models them; user turns collapse
    into one message item.
    """
    if message.role == "function":
        groups: dict[typing.Optional[str], list[MessagePart]] = {}
        for part in message.parts:
            groups.setdefault(part.function_call_id, []).append(part)
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _function_result_output(parts),
            }
            for call_id, parts in groups.items()
        ]

    if message.role == "assistant":
        keep_thinking = getattr(message, "is_live", False)
        items: list[dict] = []
        text_parts: list[dict] = []

        def _flush_text():
            if text_parts:
                items.append({"role": "assistant", "content": list(text_parts)})
                text_parts.clear()

        for part in message.parts:
            if part.type == "thinking":
                if not keep_thinking:
                    continue
                item_id, encrypted = decode_thinking_signature(part.thinking_signature)
                if not encrypted:
                    continue
                _flush_text()
                item = {
                    "type": "reasoning",
                    "id": item_id,
                    "summary": (
                        [{"type": "summary_text", "text": part.thinking}]
                        if part.thinking
                        else []
                    ),
                    "encrypted_content": encrypted,
                }
                items.append(item)
            elif part.type == "function_call":
                _flush_text()
                items.append(
                    {
                        "type": "function_call",
                        "call_id": part.function_call_id,
                        "name": part.function_call["name"],
                        "arguments": json.dumps(part.function_call["arguments"]),
                    }
                )
            elif part.type == "text":
                if part.content and part.content.strip():
                    text_parts.append({"type": "output_text", "text": part.content})
            elif part.type == "compaction":
                text_parts.append(
                    {
                        "type": "output_text",
                        "text": f"[Previous conversation summary]\n{part.content}",
                    }
                )
            else:
                # Images/documents never appear on assistant turns; be strict
                # so a mis-built history fails here rather than at the API.
                raise ValueError(f"Unsupported assistant part type: {part.type}")
        _flush_text()
        return items

    # user (and any other role) -> a single message item
    role = (
        "user" if message.role not in ("user", "system", "developer") else message.role
    )
    content = []
    for part in message.parts:
        if part.type == "thinking":
            continue
        if part.type == "function_call":
            # A user-triggered tool call has no item form; spell it out.
            content.append(
                {
                    "type": "input_text",
                    "text": f"{part.function_call['name']}:"
                    f"{json.dumps(part.function_call['arguments'])}",
                }
            )
            continue
        if part.type == "text" and (not part.content or not part.content.strip()):
            continue
        content.append(_content_part(part))
    if not content:
        return []
    return [{"role": role, "content": content}]


# ---------------------------------------------------------------------------
# Responses output -> Message
# ---------------------------------------------------------------------------


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def output_to_message(
    output: list,
    response_id: typing.Optional[str] = None,
    usage=None,
) -> Message:
    """Build the assistant Message from a response's ``output`` items."""
    parts: list[MessagePart] = []
    for item in output or []:
        item_type = _get(item, "type")
        if item_type == "reasoning":
            summary = "\n\n".join(
                _get(s, "text", "") for s in (_get(item, "summary") or [])
            )
            encrypted = _get(item, "encrypted_content")
            parts.append(
                MessagePart(
                    type="thinking",
                    thinking=summary or None,
                    thinking_signature=(
                        encode_thinking_signature(_get(item, "id"), encrypted)
                        if encrypted
                        else None
                    ),
                )
            )
        elif item_type == "message":
            text = "".join(
                _get(c, "text", "")
                for c in (_get(item, "content") or [])
                if _get(c, "type") == "output_text"
            )
            if text.strip():
                parts.append(MessagePart(type="text", content=text))
        elif item_type == "function_call":
            raw_arguments = _get(item, "arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.decoder.JSONDecodeError:
                raise FunctionCallParsingError(response_id, item)
            parts.append(
                MessagePart(
                    type="function_call",
                    function_call={"name": _get(item, "name"), "arguments": arguments},
                    function_call_id=_get(item, "call_id"),
                )
            )
        # Built-in tool items (web_search_call, ...) carry no replayable
        # state we model; their results surface in the message text.

    if not parts:
        parts.append(MessagePart(type="text", content=""))
    message = Message(role="assistant", parts=parts, id=response_id)
    message.usage = usage_to_dict(usage)
    # Produced by the API in this process: reasoning items are replayable.
    message.is_live = True
    return message


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIResponsesProvider(BaseProvider):
    def __init__(
        self,
        chat: AgentlysBase,
        model: str,
        base_url: typing.Optional[str] = None,
        api_key: typing.Optional[str] = None,
        max_tokens: typing.Optional[int] = None,
        effort: typing.Optional[str] = None,
        reasoning_summary: typing.Optional[str] = "auto",
        default_headers: typing.Optional[dict] = None,
        # Accepted for config parity with AnthropicProvider; OpenAI caches
        # prompt prefixes automatically so there is nothing to configure.
        cache_ttl: typing.Optional[str] = None,
        cache_ttl_messages: typing.Optional[str] = None,
    ):
        from openai import AsyncOpenAI

        self.chat = chat
        self.model = model
        self.max_tokens = (
            DEFAULT_MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
        )
        self.effort = resolve_effort(effort)
        self.reasoning_summary = reasoning_summary
        env_host = os.getenv("AGENTLYS_HOST")
        resolved_api_key = api_key or os.getenv("AGENTLYS_API_KEY")
        if (
            resolved_api_key is None
            and (base_url or env_host)
            and not os.getenv("OPENAI_API_KEY")
        ):
            resolved_api_key = "not-needed"
        self.client = AsyncOpenAI(
            base_url=base_url or env_host or OPENAI_DEFAULT_BASE_URL,
            api_key=resolved_api_key,
            default_headers=default_headers,
        )

    # -- request assembly ---------------------------------------------------

    def _build_instructions(self) -> typing.Optional[str]:
        blocks = [
            text
            for text in (
                self.chat.instruction,
                self.chat.context,
                self.chat.initial_tools_states,
                self.chat.tool_search_categories_hint,
            )
            if text
        ]
        return "\n\n".join(blocks) if blocks else None

    def _loaded_tool_names(self) -> set[str]:
        """Tools a tool_search result has loaded so far in this conversation.

        The Responses API has no server-side deferred loading, so the search
        tool's ``tool_references`` are honoured here: a deferred tool is sent
        only once a search has surfaced it.
        """
        loaded: set[str] = set()
        for message in self.chat.messages:
            for part in message.parts:
                if part.tool_references:
                    loaded.update(part.tool_references)
        return loaded

    def _build_tools(self) -> list[dict]:
        loaded = self._loaded_tool_names()
        tools = []
        for s in self.chat.functions_schema:
            if s.get("defer_loading") and s["name"] not in loaded:
                continue
            tool_def = {
                "type": "function",
                "name": s["name"],
                "description": s.get("description") or "No description provided",
                "parameters": s["parameters"],
            }
            if s.get("strict") is True:
                tool_def["strict"] = True
            tools.append(tool_def)
        return tools

    def _build_reasoning(self, kwargs: dict) -> typing.Optional[dict]:
        """Turn the effort / thinking settings into a ``reasoning`` block.

        Precedence: per-call ``effort`` > provider ``effort`` > per-call or
        chat-level ``thinking`` (translated). Anything already given as a
        ``reasoning`` kwarg wins outright.
        """
        thinking = kwargs.pop("thinking", None) or getattr(self.chat, "thinking", None)
        effort = resolve_effort(kwargs.pop("effort", None)) or self.effort
        if effort is None:
            effort = thinking_to_effort(thinking)
        if "reasoning" in kwargs:
            return kwargs.pop("reasoning")
        reasoning = {}
        if effort:
            reasoning["effort"] = effort
        if self.reasoning_summary:
            reasoning["summary"] = self.reasoning_summary
        return reasoning or None

    def _prepare_request_params(self, **kwargs) -> tuple[list[dict], list[dict], dict]:
        input_items: list[dict] = []
        for items in self.prepare_messages(
            transform_function=message_to_responses_items,
            transform_list_function=lambda x: add_empty_function_result(
                drop_orphaned_function_results(x)
            ),
        ):
            input_items.extend(items)

        tools = self._build_tools()

        instructions = self._build_instructions()
        if instructions and "instructions" not in kwargs:
            kwargs["instructions"] = instructions

        if self.chat.use_tools_only and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = "required"

        reasoning = self._build_reasoning(kwargs)
        if reasoning:
            kwargs["reasoning"] = reasoning

        kwargs.setdefault("store", False)
        include = list(kwargs.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        kwargs["include"] = include
        kwargs.setdefault("max_output_tokens", self.max_tokens)
        return input_items, tools, kwargs

    # -- API calls ----------------------------------------------------------

    async def fetch_async(self, **kwargs) -> Message:
        input_items, tools, kwargs = self._prepare_request_params(**kwargs)
        if tools:
            kwargs["tools"] = tools
        res = await self.client.responses.create(
            model=self.model, input=input_items, **kwargs
        )
        return output_to_message(res.output, response_id=res.id, usage=res.usage)

    async def complete(
        self,
        messages: list[dict],
        system: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict = {"store": False}
        if system:
            kwargs["instructions"] = system
        if hasattr(self, "_get_auth_headers"):
            kwargs["extra_headers"] = await self._get_auth_headers()
        res = await self.client.responses.create(
            model=model or self.model,
            input=messages,
            max_output_tokens=max_tokens,
            **kwargs,
        )
        text = getattr(res, "output_text", None) or "".join(
            _get(c, "text", "")
            for item in (res.output or [])
            if _get(item, "type") == "message"
            for c in (_get(item, "content") or [])
            if _get(c, "type") == "output_text"
        )
        if not text:
            raise RuntimeError("Completion response contained no text")
        return text

    async def fetch_stream_async(self, **kwargs):
        """Stream a response.

        Yields ``text`` and ``thinking`` deltas like the Anthropic provider,
        plus ``tool_started`` / ``tool_delta`` events for tool calls, then the
        final ``message``.
        """
        input_items, tools, kwargs = self._prepare_request_params(**kwargs)
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.responses.create(
            model=self.model, input=input_items, stream=True, **kwargs
        )

        # output_index -> {"name", "id"} for function_call items in flight
        current_tools: dict[int, dict] = {}
        final_response = None

        async for event in stream:
            event_type = _get(event, "type")
            if event_type == "response.output_text.delta":
                yield {"type": "text", "content": _get(event, "delta", "")}
            elif event_type == "response.reasoning_summary_text.delta":
                yield {"type": "thinking", "content": _get(event, "delta", "")}
            elif event_type == "response.output_item.added":
                item = _get(event, "item")
                if _get(item, "type") == "function_call":
                    index = _get(event, "output_index", 0)
                    info = {"name": _get(item, "name"), "id": _get(item, "call_id")}
                    current_tools[index] = info
                    yield {
                        "type": "tool_started",
                        "name": info["name"],
                        "id": info["id"],
                        "index": index,
                    }
            elif event_type == "response.function_call_arguments.delta":
                info = current_tools.get(_get(event, "output_index", 0), {})
                yield {
                    "type": "tool_delta",
                    "name": info.get("name"),
                    "id": info.get("id"),
                    "partial_json": _get(event, "delta", ""),
                }
            elif event_type == "response.completed":
                final_response = _get(event, "response")
            elif event_type in ("response.failed", "response.incomplete"):
                final_response = _get(event, "response")
            elif event_type == "error":
                raise RuntimeError(
                    f"OpenAI stream error: {_get(event, 'message') or event}"
                )

        if final_response is None:
            raise RuntimeError("Stream ended without a completed response")
        error = _get(final_response, "error")
        if error:
            raise RuntimeError(
                f"OpenAI response failed: {_get(error, 'message') or error}"
            )

        final_message = output_to_message(
            _get(final_response, "output"),
            response_id=_get(final_response, "id"),
            usage=_get(final_response, "usage"),
        )
        yield {"type": "message", "message": final_message}
