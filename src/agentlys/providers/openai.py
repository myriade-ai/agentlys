import json
import os
import typing

from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider
from agentlys.providers.utils import (
    FunctionCallParsingError,
    add_empty_function_result,
    drop_orphaned_function_results,
)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# reasoning effort values the API accepts. Availability varies per model
# generation (``minimal`` is gpt-5 only, ``none`` needs gpt-5.1+, ``xhigh``
# gpt-5.2+); the provider forwards whatever it is given and lets the API
# reject a value the target model does not support.
_VALID_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

# Anthropic ``output_config.effort`` levels that have no OpenAI namesake.
_EFFORT_ALIASES = {"max": "xhigh"}


def resolve_effort(value: typing.Optional[str]) -> typing.Optional[str]:
    """Normalize an effort level to what OpenAI accepts.

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

    ``enabled`` scales with the budget the caller had in mind. ``adaptive``
    and ``disabled`` send nothing: the former is the API's own default, and
    for the latter any explicit effort would be a 400 on a model without
    reasoning (gpt-4o, an Ollama model, ...).
    """
    if not thinking or thinking.get("type") != "enabled":
        return None
    budget = thinking.get("budget_tokens") or 0
    if budget < 2048:
        return "low"
    if budget < 8192:
        return "medium"
    return "high"


def create_openai_client(
    base_url: typing.Optional[str] = None,
    api_key: typing.Optional[str] = None,
    host_suffix: str = "",
    default_headers: typing.Optional[dict] = None,
):
    """Build an AsyncOpenAI client for OpenAI or any OpenAI-compatible API.

    Resolution order:
    - base_url: explicit argument > AGENTLYS_HOST env (+ host_suffix) > OpenAI
    - api_key: explicit argument > AGENTLYS_API_KEY env > OPENAI_API_KEY env

    When a custom endpoint is configured but no API key is available, a
    placeholder key is used so key-less OpenAI-compatible servers
    (Ollama, vLLM, LiteLLM, ...) work out of the box — the SDK refuses to
    build a client without a key.
    """
    from openai import AsyncOpenAI

    env_host = os.getenv("AGENTLYS_HOST")
    resolved_base_url = base_url or (f"{env_host}{host_suffix}" if env_host else None)
    resolved_api_key = api_key or os.getenv("AGENTLYS_API_KEY")
    if (
        resolved_api_key is None
        and resolved_base_url is not None
        and not os.getenv("OPENAI_API_KEY")
    ):
        resolved_api_key = "not-needed"

    return AsyncOpenAI(
        base_url=resolved_base_url or OPENAI_DEFAULT_BASE_URL,
        api_key=resolved_api_key,
        default_headers=default_headers,
    )


def usage_to_dict(usage) -> typing.Optional[dict]:
    """Normalize OpenAI usage to the naming used by Message.

    OpenAI-compatible APIs report ``prompt_tokens`` as the *total* prompt size,
    with cached and cache-written tokens as subsets of it
    (``ordinary = prompt_tokens - cached_tokens - cache_write_tokens``), while
    Anthropic reports ``input_tokens`` as the uncached remainder
    (``total = input_tokens + cache_read + cache_creation``). We subtract both
    subsets so the two providers share one shape — the shape
    ``compaction.should_compact`` already sums.
    """
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)

    def _detail(name: str) -> int:
        if details is None:
            return 0
        value = (
            details.get(name)
            if isinstance(details, dict)
            else getattr(details, name, None)
        )
        return value or 0

    # Cached reads are billed at a reduced rate, cache writes at a premium:
    # keeping them separate is what makes cost attribution possible.
    cache_read = _detail("cached_tokens")
    cache_creation = _detail("cache_write_tokens")

    # DeepSeek does not use prompt_tokens_details: it reports the same split as
    # top-level fields, with the same invariant
    # (prompt_tokens == prompt_cache_hit_tokens + prompt_cache_miss_tokens).
    if not cache_read:
        cache_read = getattr(usage, "prompt_cache_hit_tokens", None) or 0

    # Guard against providers reporting subsets larger than the total.
    cache_read = min(cache_read, prompt_tokens)
    cache_creation = min(cache_creation, prompt_tokens - cache_read)

    result = {
        "input_tokens": prompt_tokens - cache_read - cache_creation,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    if cache_read:
        result["cache_read_input_tokens"] = cache_read
    if cache_creation:
        result["cache_creation_input_tokens"] = cache_creation
    # Reasoning models report their hidden thinking as a subset of
    # completion_tokens; keep it visible for cost attribution.
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning = None
    if completion_details is not None:
        reasoning = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else getattr(completion_details, "reasoning_tokens", None)
        )
    if reasoning:
        result["reasoning_tokens"] = reasoning
    return result


def from_openai_object(
    role: str,
    content: str,
    tool_calls: typing.Optional[list] = None,
    id: typing.Optional[str] = None,
    usage: typing.Optional[dict] = None,
):
    # We need to unquote the arguments

    parts = []
    if content:
        parts.append(MessagePart(type="text", content=content))

    for tool_call in tool_calls or []:
        if tool_call.type != "function":
            raise ValueError(
                "We don't support tool calls with type other than function"
            )
        function_call = tool_call.function
        try:
            arguments = json.loads(function_call.arguments or "{}")
        except json.decoder.JSONDecodeError:
            raise FunctionCallParsingError(id, function_call)
        parts.append(
            MessagePart(
                type="function_call",
                function_call={
                    "name": function_call.name,
                    "arguments": arguments,
                },
                function_call_id=tool_call.id,
            )
        )

    if parts:
        return Message(role=role, parts=parts, id=id, usage=usage)
    return Message(role=role, content=content, id=id, usage=usage)


def build_system_messages(chat: AgentlysBase) -> list[Message]:
    """System prompt as role="system" Messages (instruction, context, tool states)."""
    return [
        Message(role="system", content=text)
        for text in (
            chat.instruction,
            chat.context,
            chat.initial_tools_states,
            chat.tool_search_categories_hint,
        )
        if text
    ]


def parts_to_openai_dict(part: MessagePart) -> dict:
    if part.type == "text":
        return {
            "type": "text",
            "text": part.content,
        }
    elif part.type == "image":
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{part.image.format};base64,{part.image.to_base64()}"
            },
        }
    elif part.type == "function_call":
        return {
            "name": part.function_call["name"],
            "arguments": json.dumps(part.function_call["arguments"]),
        }
    elif part.type == "function_result":
        return {
            "type": "text",
            "text": part.content,
        }
    elif part.type == "function_result_image":
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{part.image.format};base64,{part.image.to_base64()}"
            },
        }
    elif part.type == "document":
        if part.document is None:
            raise ValueError("Document part must have a document")
        doc = part.document
        if doc.media_type == "text/plain":
            name = doc.name or "document"
            text = doc.data.decode("utf-8", errors="replace")
            return {"type": "text", "text": f"[Document: {name}]\n{text}"}
        if doc.media_type == "application/pdf":
            return {
                "type": "file",
                "file": {
                    "filename": doc.name or "document.pdf",
                    "file_data": f"data:application/pdf;base64,{doc.to_base64()}",
                },
            }
        raise ValueError(
            f"Unsupported document media_type {doc.media_type!r}: the Chat "
            "Completions API accepts PDF files and inline text. Convert other "
            "formats first."
        )
    elif part.type == "compaction":
        return {
            "type": "text",
            "text": f"[Previous conversation summary]\n{part.content}",
        }

    raise ValueError(f"Unknown part type: {part.type}")


def message_to_openai_dict(message: Message) -> dict:
    if message.role == "function":
        res = {
            "role": "tool",
            "tool_call_id": message.function_call_id,
            "content": [parts_to_openai_dict(part) for part in message.parts],
        }
        if message.name:
            res["name"] = message.name
    else:
        res = {"role": message.role, "content": []}
        for part in message.parts:
            if part.type == "thinking":
                # Chat Completions has no reasoning-state item.
                continue
            if part.type == "function_call" and message.role == "assistant":
                res.setdefault("tool_calls", []).append(
                    {
                        "id": part.function_call_id,
                        "type": "function",
                        "function": parts_to_openai_dict(part),
                    }
                )
            elif part.type == "function_call" and message.role == "user":
                # Workaround: If the user is triggering a function, we add it's name and arguments to the content
                res["content"] = (
                    part.function_call["name"]
                    + ":"
                    + json.dumps(part.function_call["arguments"])
                )
            else:
                res["content"].append(parts_to_openai_dict(part))

        if "content" in res and len(res["content"]) == 0:
            res["content"] = None

    return res


def return_image_as_user_message(messages: list[Message]) -> list[Message]:
    """
    Adaptation because OpenAI doesn't support image in function call.
    We return the image as a user message.

    Builds new Message objects instead of mutating: the input list holds the
    same references as chat.messages, and this runs on every request.
    """
    result = []
    for message in messages:
        if message.role == "function" and message.image is not None:
            message = Message(
                role="user",
                name=message.name,
                id=message.id,
                parts=message.parts,
            )
        result.append(message)
    return result


def split_function_results(messages: list[Message]) -> list[Message]:
    """Split parallel tool results into one function message per tool call.

    OpenAI-compatible APIs require one ``role="tool"`` message per
    ``tool_call_id``, while agentlys combines parallel tool results into a
    single function message (the Anthropic convention).
    """
    result = []
    for message in messages:
        call_ids = {
            part.function_call_id
            for part in message.parts
            if part.function_call_id is not None
        }
        if message.role != "function" or len(call_ids) <= 1:
            result.append(message)
            continue
        # Preserve part order while grouping by function_call_id
        groups: dict[str, list[MessagePart]] = {}
        for part in message.parts:
            groups.setdefault(part.function_call_id, []).append(part)
        for parts in groups.values():
            result.append(Message(role="function", name=message.name, parts=parts))
    return result


class OpenAIProvider(BaseProvider):
    # Wire-format hook: subclasses can swap the message serializer
    # (see DefaultProvider's string-only variant).
    message_transform = staticmethod(message_to_openai_dict)

    def __init__(
        self,
        chat: AgentlysBase,
        model: str,
        base_url: str = None,
        api_key: str = None,
        effort: typing.Optional[str] = None,
        # Accepted for config parity with AnthropicProvider: OpenAI-compatible
        # APIs cache prompt prefixes on their own, there is nothing to set.
        cache_ttl: typing.Optional[str] = None,
        cache_ttl_messages: typing.Optional[str] = None,
    ):
        self.chat = chat
        self.model = model
        self.effort = resolve_effort(effort)
        self.client = create_openai_client(base_url=base_url, api_key=api_key)

    def _loaded_tool_names(self) -> set[str]:
        """Tools a tool_search result has loaded so far in this conversation.

        OpenAI-compatible APIs have no server-side deferred loading, so the
        search tool's ``tool_references`` are honoured here: a deferred tool
        is sent only once a search has surfaced it.
        """
        loaded: set[str] = set()
        for message in self.chat.messages:
            for part in message.parts:
                if part.tool_references:
                    loaded.update(part.tool_references)
        return loaded

    def _apply_reasoning_effort(self, kwargs: dict) -> None:
        """Translate the effort / thinking settings into ``reasoning_effort``.

        Per-call ``effort`` beats the provider default, which beats a
        translated Anthropic-style ``thinking`` config. ``thinking`` itself
        is never forwarded: the API rejects unknown parameters.
        """
        thinking = kwargs.pop("thinking", None) or getattr(self.chat, "thinking", None)
        effort = resolve_effort(kwargs.pop("effort", None)) or getattr(
            self, "effort", None
        )
        if effort is None:
            effort = thinking_to_effort(thinking)
        if effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = effort

    def _prepare_request_params(self, **kwargs):
        """Prepare messages, tools, and kwargs for an OpenAI-compatible request."""
        messages = self.prepare_messages(
            transform_function=self.message_transform,
            transform_list_function=lambda x: split_function_results(
                add_empty_function_result(
                    return_image_as_user_message(drop_orphaned_function_results(x))
                )
            ),
        )

        # An assistant turn holding nothing but a thinking block serializes
        # to content=None with no tool_calls, which the API rejects.
        messages = [
            m
            for m in messages
            if not (
                m.get("role") == "assistant"
                and not m.get("content")
                and not m.get("tool_calls")
            )
        ]
        system_messages = build_system_messages(self.chat)
        messages = [self.message_transform(sm) for sm in system_messages] + messages

        if self.chat.use_tools_only and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = "required"

        self._apply_reasoning_effort(kwargs)

        tools = []
        if self.chat.functions_schema:
            loaded = self._loaded_tool_names()
            for tool_schema in self.chat.functions_schema:
                if (
                    tool_schema.get("defer_loading")
                    and tool_schema["name"] not in loaded
                ):
                    continue
                # defer_loading is an agentlys-level flag, not an API field.
                clean_schema = {
                    k: v for k, v in tool_schema.items() if k != "defer_loading"
                }
                tools.append({"type": "function", "function": clean_schema})

        return messages, tools, kwargs

    async def fetch_async(self, **kwargs) -> Message:
        messages, tools, kwargs = self._prepare_request_params(**kwargs)

        if tools:
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        else:
            res = await self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )

        message = res.choices[0].message
        return from_openai_object(
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            id=res.id,  # We use the response id as the message id
            usage=usage_to_dict(res.usage),
        )

    async def complete(
        self,
        messages: list[dict],
        system: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages
        kwargs = {}
        # If the provider exposes auth headers (e.g. proxy), inject them
        if hasattr(self, "_get_auth_headers"):
            kwargs["extra_headers"] = await self._get_auth_headers()

        # max_tokens is rejected by reasoning models (o-series, gpt-5);
        # max_completion_tokens by some OpenAI-compatible gateways.
        # AGENTLYS_OPENAI_LEGACY_MAX_TOKENS=1 keeps the old field for those.
        if os.getenv("AGENTLYS_OPENAI_LEGACY_MAX_TOKENS") == "1":
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_completion_tokens"] = max_tokens
        res = await self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            **kwargs,
        )
        content = res.choices[0].message.content
        if not content:
            raise RuntimeError("Completion response contained no text")
        return content

    async def fetch_stream_async(self, **kwargs):
        """Stream response tokens from any OpenAI-compatible chat completions API.

        Yields text chunks as they arrive, then the final Message
        (with potential tool calls) after streaming completes.
        """
        messages, tools, kwargs = self._prepare_request_params(**kwargs)
        if tools:
            kwargs["tools"] = tools
        # Streamed responses omit usage unless asked; without it the final
        # Message has no token counts and compaction never triggers.
        # AGENTLYS_OPENAI_STREAM_USAGE=0 opts out for gateways that reject
        # stream_options.
        if os.getenv("AGENTLYS_OPENAI_STREAM_USAGE", "1") != "0":
            kwargs.setdefault("stream_options", {"include_usage": True})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        response_id = None
        role = "assistant"
        content_chunks: list[str] = []
        # index -> accumulated tool call (OpenAI streams tool calls in fragments)
        tool_calls: dict[int, dict] = {}
        usage = None

        async for chunk in stream:
            if response_id is None:
                response_id = chunk.id
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.role:
                role = delta.role
            if delta.content:
                content_chunks.append(delta.content)
                yield {"type": "text", "content": delta.content}
            for tool_call in delta.tool_calls or []:
                entry = tool_calls.setdefault(
                    tool_call.index, {"id": None, "name": "", "arguments": ""}
                )
                if tool_call.id:
                    entry["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        entry["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        entry["arguments"] += tool_call.function.arguments

        parts = []
        content = "".join(content_chunks)
        if content:
            parts.append(MessagePart(type="text", content=content))
        for index in sorted(tool_calls):
            entry = tool_calls[index]
            try:
                arguments = json.loads(entry["arguments"] or "{}")
            except json.decoder.JSONDecodeError:
                raise FunctionCallParsingError(response_id, entry)
            parts.append(
                MessagePart(
                    type="function_call",
                    function_call={
                        "name": entry["name"],
                        "arguments": arguments,
                    },
                    function_call_id=entry["id"],
                )
            )

        final_message = Message(
            role=role, parts=parts, id=response_id, usage=usage_to_dict(usage)
        )
        yield {"type": "message", "message": final_message}
