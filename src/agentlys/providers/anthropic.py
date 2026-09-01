import hashlib
import json
import logging
import os

import anthropic
from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider
from agentlys.providers.utils import (
    add_empty_function_result,
    drop_orphaned_function_results,
)

logger = logging.getLogger(__name__)

# Prompt-cache TTLs supported by the API. "5m" is the API default and is
# emitted as a bare {"type": "ephemeral"} so requests stay byte-identical to
# the pre-TTL behaviour when nothing is configured.
_VALID_CACHE_TTLS = ("5m", "1h")
_EXTENDED_TTL_BETA = "extended-cache-ttl-2025-04-11"

# output_config.effort levels. Unset means "don't send", i.e. the model's
# own default.
_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Anthropic's server-side tool search (BM25 variant: natural-language
# queries). Injected into `tools` when the chat enables server-side tool
# search; deferred tools are then discovered and expanded by the API itself,
# with no client round-trip. GA — no beta header required.
SERVER_TOOL_SEARCH_DEF = {
    "type": "tool_search_tool_bm25_20251119",
    "name": "tool_search_tool_bm25",
}


def _resolve_cache_ttl(value: str | None, env_var: str, default: str) -> str:
    ttl = value or os.getenv(env_var) or default
    if ttl not in _VALID_CACHE_TTLS:
        raise ValueError(
            f"Invalid cache TTL {ttl!r} (from {env_var} or constructor): "
            f"expected one of {_VALID_CACHE_TTLS}"
        )
    return ttl


def _cache_control(ttl: str) -> dict:
    """Build a cache_control block for the given TTL.

    The 5-minute TTL is the API default, so it is emitted without an explicit
    ``ttl`` key — that keeps the request bytes identical to what previous
    versions sent, and to what any already-warm cache entry was written with.
    """
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _resolve_effort(value: str | None) -> str | None:
    effort = value or os.getenv("AGENTLYS_EFFORT") or None
    if effort is not None and effort not in _VALID_EFFORTS:
        raise ValueError(f"Invalid effort {effort!r}: expected one of {_VALID_EFFORTS}")
    return effort


def _sha(payload) -> str:
    dump = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()[:16]


def _canonical_key_order(value):
    """Rebuild dicts with sorted keys, recursively.  List order is preserved."""
    if isinstance(value, dict):
        return {key: _canonical_key_order(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_key_order(item) for item in value]
    return value


def part_to_anthropic_dict(part: MessagePart) -> dict:
    if part.type == "text":
        return {
            "type": "text",
            "text": part.content,
        }
    elif part.type == "image":
        if part.image is None:
            raise ValueError("Image part must have an image")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.image.format,
                "data": part.image.to_base64(),
            },
        }
    elif part.type == "document":
        if part.document is None:
            raise ValueError("Document part must have a document")
        # Document is bytes-based, which maps to two of Anthropic's document
        # sources: "base64" (whose only supported binary format is PDF) and
        # "text" (text/plain). The API also has url/content/file sources, but
        # those aren't byte payloads and aren't modeled by Document (yet).
        # Reject unsupported media types here so they fail early instead of
        # at request time. Note: on Bedrock/Vertex only base64 is available.
        if part.document.media_type == "application/pdf":
            source = {
                "type": "base64",
                "media_type": "application/pdf",
                "data": part.document.to_base64(),
            }
        elif part.document.media_type == "text/plain":
            source = {
                "type": "text",
                "media_type": "text/plain",
                "data": part.document.data.decode("utf-8"),
            }
        else:
            raise ValueError(
                f"Unsupported document media_type {part.document.media_type!r}: "
                "Anthropic accepts binary (base64) documents only as "
                "application/pdf, and inline text documents as text/plain. "
                "Convert other formats (e.g. .docx, .xlsx) to PDF or plain text."
            )
        block = {
            "type": "document",
            "source": source,
        }
        if part.document.name:
            block["title"] = part.document.name
        return block
    elif part.type == "function_call":
        return {
            "type": "tool_use",
            "id": part.function_call_id,
            "name": part.function_call["name"],
            # Canonical key order.  A caller that round-trips its history
            # through a store which reorders object keys — Postgres jsonb sorts
            # them by length then bytewise — would otherwise serialize a
            # reloaded tool call differently from the live one and lose the
            # cached prefix from that message on.  Key order carries no
            # meaning in JSON, so imposing one costs nothing.
            "input": _canonical_key_order(part.function_call["arguments"]),
        }
    elif part.type == "function_result_image":
        if part.image is None:
            raise ValueError("Function result image part must have an image")
        # Build content blocks: include text content if present, then image
        content_blocks = []
        if part.content:
            content_blocks.append(
                {
                    "type": "text",
                    "text": part.content,
                }
            )
        content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.image.format,
                    "data": part.image.to_base64(),
                },
            }
        )
        return {
            "type": "tool_result",
            "content": content_blocks,
            "tool_use_id": part.function_call_id,
        }
    elif part.type == "function_result":
        result = {
            "type": "tool_result",
            "tool_use_id": part.function_call_id,
        }
        if part.tool_references is not None:
            result["content"] = [
                {"type": "tool_reference", "tool_name": name}
                for name in part.tool_references
            ]
        else:
            result["content"] = part.content
        return result
    elif part.type in ("server_tool_use", "server_tool_result"):
        # Replay the raw API block verbatim — the API executed the tool, so
        # the client must neither rewrite it nor answer it with a
        # tool_result. Canonical key order rebuilds a fresh structure (no
        # shared mutation when cache_control is stamped on it) and keeps the
        # serialization byte-stable after a store that reorders object keys
        # (Postgres jsonb), mirroring the tool_use "input" handling above.
        return _canonical_key_order(part.function_call)
    elif part.type == "thinking":
        if part.is_redacted:
            return {
                "type": "redacted_thinking",
                "data": part.thinking_signature,
            }
        return {
            "type": "thinking",
            "thinking": part.thinking,
            "signature": part.thinking_signature,
        }
    elif part.type == "compaction":
        return {
            "type": "text",
            "text": f"[Previous conversation summary]\n{part.content}",
        }
    raise ValueError(f"Unknown part type: {part.type}")


def message_to_anthropic_dict(message: Message) -> dict:
    res = {
        "role": message.role if message.role in ["user", "assistant"] else "user",
        "content": [],
    }

    # Thinking blocks are replayed only for messages flagged is_live: those
    # this process received from the API, and those a caller restored
    # byte-for-byte through load_messages(keep_thinking=True).  Replaying them
    # is what keeps the cached prefix intact — dropping a block rewrites the
    # assistant message it belongs to, invalidating the cache from there on,
    # within a tool loop and across turns alike.  Anything else rebuilt from
    # storage is stripped: a re-serialized block may have lost its signature
    # or its position, and the docs confirm omitting thinking from prior turns
    # is accepted.
    keep_thinking = getattr(message, "is_live", False)

    for part in message.parts:
        if part.type == "text" and (not part.content or not part.content.strip()):
            continue
        if part.type == "thinking" and not keep_thinking:
            continue
        if part.type == "thinking" and not part.is_redacted and not part.thinking:
            # The model does emit signed thinking blocks with empty content,
            # but the API refuses them on replay ('each thinking block must
            # contain thinking', 400) when one sits on the last assistant
            # message — which a replayed empty turn does. Dropping the block
            # at worst invalidates the cached prefix; replaying it kills the
            # conversation. Redacted blocks carry their payload in the
            # signature and stay.
            continue
        res["content"].append(part_to_anthropic_dict(part))

    return res


DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "10000"))


class AnthropicProvider(BaseProvider):
    supports_server_tool_search = True

    def __init__(
        self,
        chat: AgentlysBase,
        model: str,
        max_tokens: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_ttl: str | None = None,
        cache_ttl_messages: str | None = None,
        effort: str | None = None,
    ):
        self.model = model
        env_host = os.getenv("AGENTLYS_HOST")
        self.client = anthropic.AsyncAnthropic(
            base_url=base_url or env_host or "https://api.anthropic.com",
            # None lets the SDK fall back to the ANTHROPIC_API_KEY env var
            api_key=api_key or os.getenv("AGENTLYS_API_KEY"),
        )
        self.chat = chat
        self.max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        # System + tools breakpoints, then the message breakpoints.  They are
        # configured separately because a 1h write costs 2x base input against
        # 1.25x for 5m: the system/tools prefix is worth the premium far more
        # often than the conversation tail is.
        self.cache_ttl = _resolve_cache_ttl(cache_ttl, "AGENTLYS_CACHE_TTL", "5m")
        self.cache_ttl_messages = _resolve_cache_ttl(
            cache_ttl_messages, "AGENTLYS_CACHE_TTL_MESSAGES", self.cache_ttl
        )
        if self.cache_ttl_messages == "1h" and self.cache_ttl == "5m":
            # The API requires longer-TTL entries to come first, and tools +
            # system are always rendered before messages.
            logger.warning(
                "AGENTLYS_CACHE_TTL_MESSAGES=1h needs AGENTLYS_CACHE_TTL=1h "
                "(longer-lived cache entries must come first); using 5m for "
                "the message breakpoints."
            )
            self.cache_ttl_messages = "5m"
        self.effort = _resolve_effort(effort)
        self.cache_debug = os.getenv("AGENTLYS_CACHE_DEBUG") == "1"

    @staticmethod
    def _merge_same_role_messages(messages: list[dict]) -> list[dict]:
        """Merge consecutive messages sharing the same role.

        The Anthropic API rejects two consecutive messages with the same
        role, e.g. a tool_result message directly followed by a user message.
        """
        merged_messages = []
        for message in messages:
            if merged_messages and merged_messages[-1]["role"] == message["role"]:
                # Convert string content to list format for merging
                if isinstance(merged_messages[-1]["content"], str):
                    merged_messages[-1]["content"] = [
                        {
                            "type": "text",
                            "text": merged_messages[-1]["content"],
                        }
                    ]

                if not isinstance(merged_messages[-1]["content"], list):
                    raise ValueError(
                        f"Invalid content type: {type(merged_messages[-1]['content'])}"
                    )

                # Track existing tool_use_ids to prevent duplicates
                existing_tool_use_ids = {
                    c.get("tool_use_id")
                    for c in merged_messages[-1]["content"]
                    if isinstance(c, dict) and c.get("type") == "tool_result"
                }
                # Only add content blocks that aren't duplicate tool_results
                for content_block in message["content"]:
                    if (
                        isinstance(content_block, dict)
                        and content_block.get("type") == "tool_result"
                    ):
                        tool_use_id = content_block.get("tool_use_id")
                        if tool_use_id in existing_tool_use_ids:
                            continue
                        existing_tool_use_ids.add(tool_use_id)
                    merged_messages[-1]["content"].append(content_block)
            else:
                merged_messages.append(message)
        return merged_messages

    def _build_tools(self) -> list[dict]:
        """Translate function schemas to Anthropic tool definitions.

        Maps "parameters" to "input_schema" and guarantees a description.

        ``strict`` is forwarded when the caller sets it and never inferred.
        Deciding it here meant reading the model name, which is not something
        a provider can rely on knowing: behind a gateway it is a logical role
        or absent entirely, so the guess was wrong exactly where the schemas
        were most worth constraining. Closed schemas
        (``additionalProperties: false``, set in ``utils.inspect_schema``)
        already tell the model which arguments exist; ``strict`` on top of
        that is the caller's call, along with the per-request limits it has
        to stay inside.
        """
        tools = []
        for s in self.chat.functions_schema:
            tool_def = {
                "name": s["name"],
                "description": s["description"] or "No description provided",
                "input_schema": s["parameters"],
            }
            if s.get("strict") is True:
                tool_def["strict"] = True
            if s.get("defer_loading"):
                tool_def["defer_loading"] = True
            tools.append(tool_def)

        # Server-side tool search: the API runs the search itself, so no
        # local tool_search function is registered (see enable_tool_search).
        # The server tool leads the list and is never deferred — the API
        # requires at least one non-deferred tool.
        tool_search_config = getattr(self.chat, "_tool_search_config", None)
        if tool_search_config is not None and getattr(
            tool_search_config, "server_side", False
        ):
            tools.insert(0, dict(SERVER_TOOL_SEARCH_DEF))
        return tools

    def _build_system_blocks(self) -> list[dict]:
        """Assemble the system prompt blocks (instruction, context, tools).

        cache_control goes on the LAST system block so the entire system
        section is cached as a unit.  Anthropic uses cumulative hashes —
        placing cache_control on system[0] while system[1] varies
        invalidates every downstream block.
        """
        blocks = []
        for text in (
            self.chat.instruction,
            self.chat.context,
            self.chat.initial_tools_states,
            self.chat.tool_search_categories_hint,
        ):
            if text:
                blocks.append({"type": "text", "text": text})
        if blocks:
            blocks[-1]["cache_control"] = _cache_control(self.cache_ttl)
        return blocks

    def _apply_cache_control(self, messages: list[dict], tools: list[dict]) -> None:
        """Place cache_control breakpoints on messages and tools (in place).

        Anthropic's prompt caching uses up to 4 breakpoints per request.
        We use all 4: system[-1] (see _build_system_blocks), tools[-1],
        messages[-3], messages[-1].

        Why messages[-3]?  Each tool-loop iteration appends exactly 2
        messages (1 assistant + 1 tool_result).  So messages[-3] in the
        current call corresponds to messages[-1] from the *previous* call,
        whose prefix was already cached.  By keeping a breakpoint there,
        Anthropic can serve that prefix from cache_read instead of
        re-caching the entire message history on every iteration.
        """

        message_cache_control = _cache_control(self.cache_ttl_messages)

        def _set_cache_control(msg_index):
            """Add cache_control to the last content block of messages[msg_index]."""
            msg = messages[msg_index]
            if (
                isinstance(msg["content"], list)
                and len(msg["content"]) > 0
                and isinstance(msg["content"][-1], dict)
            ):
                msg["content"][-1]["cache_control"] = dict(message_cache_control)
            elif isinstance(msg["content"], str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": dict(message_cache_control),
                    }
                ]

        if messages:
            # Breakpoint on messages[-1]: caches the full conversation
            _set_cache_control(-1)

            # Breakpoint on messages[-3]: retains the previous iteration's
            # cache so Anthropic can read from it (cache_read) instead of
            # re-caching the entire prefix (cache_creation).
            if len(messages) >= 3:
                _set_cache_control(-3)

        # Tools: Add cache_control to the last tool function.
        # Tools with defer_loading=true cannot carry cache_control, so
        # walk backward to the last eligible tool.
        for tool in reversed(tools):
            if not tool.get("defer_loading"):
                tool["cache_control"] = _cache_control(self.cache_ttl)
                break

    def _apply_extended_ttl_beta(self, kwargs: dict) -> None:
        """Advertise the 1h-cache beta on the direct Anthropic path.

        The 1-hour TTL is GA, but the beta flag is still accepted and is
        required by older gateways.  Set AGENTLYS_CACHE_TTL_BETA=0 to skip it
        (e.g. a proxy that rejects unknown betas).  Existing extra_headers —
        the auth headers a proxy provider injects — are preserved.
        """
        if "1h" not in (self.cache_ttl, self.cache_ttl_messages):
            return
        if os.getenv("AGENTLYS_CACHE_TTL_BETA", "1") == "0":
            return
        headers = dict(kwargs.get("extra_headers") or {})
        existing = headers.get("anthropic-beta", "")
        betas = [b.strip() for b in existing.split(",") if b.strip()]
        if _EXTENDED_TTL_BETA not in betas:
            betas.append(_EXTENDED_TTL_BETA)
        headers["anthropic-beta"] = ",".join(betas)
        kwargs["extra_headers"] = headers

    def _apply_effort(self, kwargs: dict) -> None:
        """Send output_config.effort when configured.

        Passed through extra_body so it works with SDK versions that predate
        the typed `output_config` parameter.  A per-call ``effort=`` kwarg
        (e.g. a higher effort for the final answer) wins over the provider
        default; unset means the field is not sent at all.
        """
        effort = _resolve_effort(kwargs.pop("effort", None)) or self.effort
        if not effort:
            return
        extra_body = dict(kwargs.get("extra_body") or {})
        output_config = dict(extra_body.get("output_config") or {})
        output_config.setdefault("effort", effort)
        extra_body["output_config"] = output_config
        kwargs["extra_body"] = extra_body

    def _log_cache_debug(self, system_blocks, tools, messages) -> None:
        """Log per-section hashes so prefix drift is visible without a patch."""
        logger.info(
            "agentlys cache request: system=%s tools=%s messages=%s",
            _sha(system_blocks),
            _sha(tools),
            [_sha(m) for m in messages],
        )

    def _log_cache_usage(self, usage) -> None:
        if not self.cache_debug or not usage:
            return
        logger.info(
            "agentlys cache usage: %s",
            {k: v for k, v in usage.items() if "token" in k},
        )

    def _prepare_request_params(self, **kwargs):
        """Prepare messages, tools, and kwargs for Anthropic API request."""
        messages = self.prepare_messages(
            transform_function=message_to_anthropic_dict,
            transform_list_function=lambda x: add_empty_function_result(
                drop_orphaned_function_results(x)
            ),
        )
        # A message whose parts were all filtered out (an assistant turn
        # holding nothing but a non-replayable thinking block, say) would go
        # out as content=[], which the API rejects. Drop it before merging so
        # its neighbours can still collapse into a single turn.
        messages = [m for m in messages if m["content"]]
        messages = self._merge_same_role_messages(messages)

        tools = self._build_tools()
        self._apply_cache_control(messages, tools)

        system_blocks = self._build_system_blocks()
        if system_blocks:
            kwargs["system"] = system_blocks

        if self.chat.use_tools_only and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = {"type": "any"}

        # Add thinking config if set at class level and not already in kwargs
        if getattr(self.chat, "thinking", None) and "thinking" not in kwargs:
            kwargs["thinking"] = self.chat.thinking

        self._apply_effort(kwargs)
        self._apply_extended_ttl_beta(kwargs)

        if self.cache_debug:
            self._log_cache_debug(system_blocks, tools, messages)

        return messages, tools, kwargs

    async def fetch_async(self, **kwargs):
        messages, tools, kwargs = self._prepare_request_params(**kwargs)

        res = await self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        res_dict = res.to_dict()
        msg = Message.from_anthropic_dict(
            role=res_dict["role"],
            content=res_dict["content"],
        )
        msg.usage = res_dict.get("usage")
        self._log_cache_usage(msg.usage)
        return msg

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        kwargs = {}
        if system:
            kwargs["system"] = system
        # If the provider exposes auth headers (e.g. proxy), inject them
        if hasattr(self, "_get_auth_headers"):
            kwargs["extra_headers"] = await self._get_auth_headers()

        response = await self.client.messages.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        # Extract the text (skip ThinkingBlocks when extended thinking is enabled)
        text_block = next(
            (block for block in response.content if block.type == "text"), None
        )
        if text_block is None:
            raise RuntimeError("Completion response contained no text block")
        return text_block.text

    async def fetch_stream_async(self, **kwargs):
        """Stream response tokens from Anthropic.

        Yields text chunks as they arrive. Returns the final Message
        (with potential tool calls) after streaming completes.
        """
        messages, tools, kwargs = self._prepare_request_params(**kwargs)

        async with self.client.messages.stream(
            model=self.model,
            messages=messages,
            tools=tools if tools else anthropic.NOT_GIVEN,
            max_tokens=self.max_tokens,
            **kwargs,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        yield {"type": "thinking", "content": event.delta.thinking}
                    elif event.delta.type == "text_delta":
                        yield {"type": "text", "content": event.delta.text}

            # Get final message for tool handling
            response = await stream.get_final_message()
            res_dict = response.to_dict()
            final_message = Message.from_anthropic_dict(
                role=res_dict["role"],
                content=res_dict["content"],
            )
            final_message.usage = res_dict.get("usage")
            self._log_cache_usage(final_message.usage)
            yield {"type": "message", "message": final_message}
