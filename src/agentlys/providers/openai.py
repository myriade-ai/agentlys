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


def create_openai_client(
    base_url: typing.Optional[str] = None,
    api_key: typing.Optional[str] = None,
    host_suffix: str = "",
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
    )


def usage_to_dict(usage) -> typing.Optional[dict]:
    """Normalize OpenAI usage to the input/output_tokens naming used by Message."""
    if usage is None:
        return None
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


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
        for text in (chat.instruction, chat.context, chat.initial_tools_states)
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
    ):
        self.chat = chat
        self.model = model
        self.client = create_openai_client(base_url=base_url, api_key=api_key)

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

        system_messages = build_system_messages(self.chat)
        messages = [self.message_transform(sm) for sm in system_messages] + messages

        if self.chat.use_tools_only and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = "required"

        tools = []
        if self.chat.functions_schema:
            for tool_schema in self.chat.functions_schema:
                # Strip defer_loading from the function schema before sending
                clean_schema = {
                    k: v for k, v in tool_schema.items() if k != "defer_loading"
                }
                tool_def = {
                    "type": "function",
                    "function": clean_schema,
                }
                if tool_schema.get("defer_loading"):
                    tool_def["defer_loading"] = True
                tools.append(tool_def)

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

        res = await self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
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
