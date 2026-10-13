"""Agentlys chat loop: agent state, tool registration and conversation runs."""

import asyncio
import contextvars
import copy
import inspect
import io
import json
import logging
import os
import threading
import traceback
import typing
import uuid
import warnings
import weakref
from typing import TYPE_CHECKING, Type

from PIL import Image as PILImage

from agentlys.base import AgentlysBase
from agentlys.model import Image, Message, MessagePart
from agentlys.providers.base_provider import (
    APIProvider,  # noqa: F401  (re-exported via agentlys.__init__)
    BaseProvider,
)
from agentlys.providers.utils import FunctionCallParsingError, get_provider_and_model
from agentlys.utils import (
    csv_dumps,
    get_event_loop_or_create,
    inspect_schema,
    parse_chat_template,
    truncate_with_marker,
)

if TYPE_CHECKING:
    from agentlys.compaction import CompactionHandler

AGENTLYS_MODEL = os.getenv("AGENTLYS_MODEL")
OUTPUT_SIZE_LIMIT = int(os.getenv("AGENTLYS_OUTPUT_SIZE_LIMIT", 20_000))

DEFAULT_COMPUTE_LEVELS = {
    "high": "claude-opus-4-6",
    "medium": "claude-sonnet-4-6",
    "low": "claude-haiku-4-5-20251001",
}


def _resolve_thinking_for_model(
    thinking: typing.Optional[dict],
    target_model: str,
) -> typing.Optional[dict]:
    """Adjust thinking config for model compatibility when switching models.

    - Opus/Sonnet: prefer adaptive thinking
    - Haiku: only supports extended thinking (type=enabled with budget_tokens)
    """
    if not thinking:
        return thinking

    thinking_type = thinking.get("type")
    model_lower = target_model.lower()

    if "haiku" in model_lower:
        if thinking_type == "adaptive":
            return {
                "type": "enabled",
                "budget_tokens": thinking.get("budget_tokens", 5000),
            }
        return thinking

    if "opus" in model_lower or "sonnet" in model_lower:
        if thinking_type == "enabled":
            return {"type": "adaptive"}
        return thinking

    return thinking


_from_response_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _accepts_from_response(func) -> bool:
    """Whether the callable declares a ``from_response`` parameter.

    Cached per underlying function: signature reflection on every tool
    invocation adds up in long agent loops.
    """
    target = getattr(func, "__func__", func)
    try:
        return _from_response_cache[target]
    except KeyError:
        pass
    except TypeError:
        # Not weak-referenceable/hashable — fall back to direct inspection
        return "from_response" in inspect.signature(func).parameters
    result = "from_response" in inspect.signature(func).parameters
    try:
        _from_response_cache[target] = result
    except TypeError:
        pass
    return result


# One capture of tool states at a time per event loop.  Agents that know
# nothing of each other are often built on the same host state (two sub-agents
# on one database session), and the captures of one loop could not overlap
# while they ran on it: hosts rely on that.  Per loop, not per process: a
# __llm__() may wait for a turn that runs on another loop (an agent on a
# helper thread), whose capture must not wait for it in return.  The lock is
# taken on the worker thread, so a capture that waits never holds the loop.
# Re-entrant, for the states computed inline on the loop thread (see
# initial_tools_states): a __llm__() called there may read another agent's.
_tools_states_locks = weakref.WeakKeyDictionary()


def _tools_states_lock():
    """The lock shared by the captures started from the running event loop."""
    loop = asyncio.get_running_loop()
    lock = _tools_states_locks.get(loop)
    if lock is None:
        lock = _tools_states_locks.setdefault(loop, threading.RLock())
    return lock


async def _run_in_thread(func, *args):
    """Run blocking host code on a worker thread and wait for it to finish.

    A thread cannot be interrupted.  If the task awaiting it were simply
    cancelled, the code would keep running behind its back — still reading
    state (a database session, say) that the host closes once it has seen the
    cancellation.  So a cancellation that arrives while the code is running is
    held back until it has finished, then raised; the event loop stays free
    in the meantime.  Code still waiting for a thread is never started.
    """
    # Taken by whichever comes first: the thread starting, or the cancellation.
    claim = threading.Lock()

    def run():
        if claim.acquire(blocking=False):
            return func(*args)

    # What asyncio.to_thread() does, without a task around it: a loop that
    # closes cancels every task, and a cancelled task reads as done while its
    # thread is still running.  Nothing cancels this future.
    worker = asyncio.get_running_loop().run_in_executor(
        None, contextvars.copy_context().run, run
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if not claim.acquire(blocking=False):
            while not worker.done():
                try:
                    await asyncio.wait([worker])
                except asyncio.CancelledError:
                    pass
            if not worker.cancelled():
                # Retrieved so it is not logged as lost: the cancellation wins.
                worker.exception()
        raise


# One image is encoded at a time.  Two agents can hold the same image (one
# history handed to several of them), each encodes it on its own worker
# thread, and PIL does not support two threads working on one image.
_image_encoding_lock = threading.Lock()


class _SerializedDict(typing.NamedTuple):
    """The JSON text of a dict tool result, as far as it was built."""

    head: str  # its beginning: all of it, or just past the output limit
    length: int  # the size of all of it


async def _serialize_dict(content: dict) -> _SerializedDict:
    """``json.dumps(content, default=str)``, built no further than is shown.

    The model reads at most OUTPUT_SIZE_LIMIT characters of a result.  Dumping
    a large one whole on the event loop freezes the loop to throw nearly all
    of it away, and json holds the GIL, so a thread would not help.  The text
    is produced piece by piece instead: the loop stops at the limit, and when
    there is more, a worker thread goes through the rest only to measure it —
    the truncation marker still says how much was dropped.
    """
    # default=str: tool results routinely carry values json doesn't
    # know (Decimal, datetime, UUID, ...); stringify instead of crashing
    pieces = json.JSONEncoder(default=str).iterencode(content)
    head, length = [], 0
    for piece in pieces:
        head.append(piece)
        length += len(piece)
        if length > OUTPUT_SIZE_LIMIT:
            length += await _run_in_thread(lambda: sum(map(len, pieces)))
            break
    return _SerializedDict("".join(head), length)


class StopLoopException(Exception):
    pass


def simple_response_default_callback(response: Message) -> Message:
    """
    This function is called when the response is a simple response (no function call).
    By default, the conversation will stop after a simple response.
    You can override this function to change this behavior.
    It is called on the event loop: it must not block.
    """
    raise StopLoopException("Stopping the conversation after a simple response")


class Agentlys(AgentlysBase):
    def __llm__(self):
        return self.name  # If an agent is used as a tool, it must have a name

    def __init__(
        self,
        name: str = None,
        instruction: str = None,
        examples: typing.Union[list[Message], None] = None,
        messages: typing.Union[list[Message], None] = None,
        context: str = None,
        user_context: str = None,
        max_interactions: int = 100,
        model=AGENTLYS_MODEL,
        provider: typing.Union[str, Type[BaseProvider], None] = None,
        use_tools_only: bool = False,
        mcp_servers: typing.Union[list[object], None] = None,
        thinking: typing.Optional[dict] = None,
        compaction: typing.Optional["CompactionHandler"] = None,
        cancel_event: typing.Optional[asyncio.Event] = None,
        base_url: typing.Optional[str] = None,
        api_key: typing.Optional[str] = None,
        cache_ttl: typing.Optional[str] = None,
        cache_ttl_messages: typing.Optional[str] = None,
        effort: typing.Optional[str] = None,
    ) -> None:
        """
        Initialize the Agentlys instance.
        Args:
            provider: Union[str, Type[BaseProvider], None] = None,
                Provider name ("openai", "anthropic", ...) or a custom
                BaseProvider subclass. Defaults to the AGENTLYS_PROVIDER
                environment variable, then "openai".
            use_tools_only: bool = False,
                If True, the chat will only use tools and not the LLM.
                This is a beta feature and may change in the future.
            thinking: Optional[dict] = None,
                Extended thinking configuration for Anthropic models.
                Example: {"type": "enabled", "budget_tokens": 10000}
            base_url: Optional[str] = None,
                Custom API endpoint. With provider="openai", any
                OpenAI-compatible API can be used (Ollama, vLLM, LiteLLM,
                OpenRouter, ...), e.g. "http://localhost:11434/v1".
                Defaults to the AGENTLYS_HOST environment variable.
            api_key: Optional[str] = None,
                API key for the endpoint. Defaults to the AGENTLYS_API_KEY
                environment variable, then the provider's own env var
                (OPENAI_API_KEY / ANTHROPIC_API_KEY).
            cache_ttl: Optional[str] = None,
                Anthropic prompt-cache TTL for the system + tools
                breakpoints: "5m" (default) or "1h". Defaults to the
                AGENTLYS_CACHE_TTL environment variable.
            cache_ttl_messages: Optional[str] = None,
                Same, for the message breakpoints (a 1h write costs 2x base
                input against 1.25x for 5m, so the conversation tail is
                configured separately). Defaults to
                AGENTLYS_CACHE_TTL_MESSAGES, then cache_ttl.
            effort: Optional[str] = None,
                output_config.effort: "low", "medium", "high", "xhigh" or
                "max". Defaults to the AGENTLYS_EFFORT environment variable;
                unset means the field is not sent at all.
        """
        self.provider, self.model = get_provider_and_model(
            self,
            provider,
            model,
            base_url=base_url,
            api_key=api_key,
            cache_ttl=cache_ttl,
            cache_ttl_messages=cache_ttl_messages,
            effort=effort,
        )  # TODO: rename register ?
        self.simple_response_callback = simple_response_default_callback
        if use_tools_only:
            warnings.warn(
                "use_tools_only is a beta feature and may change in the future"
            )
        self.use_tools_only = use_tools_only
        self.name = name
        self.instruction = instruction
        if examples is None:
            self.examples = []
        else:
            self.examples = examples

        if messages is None:
            self.messages = []
        else:
            self.messages = messages

        self.context = context
        self.user_context = user_context
        self.max_interactions = max_interactions
        self.thinking = thinking
        self.compaction = compaction
        self.cancel_event = cancel_event
        self._initial_tools_states = None
        self._tools_states_pinned = False
        # The message whose turn already has its snapshot (see
        # _ensure_tools_states), so its capture does not take a second one.
        self._tools_states_taken_for = None
        # Set on a sub-agent whose states its parent captured at the start of
        # its turn (capture_states_at_turn_start): its invocations keep them.
        self._tools_states_from_parent_turn = False
        self.functions_schema = []
        self.functions = {}
        self.tools = {}
        self._sub_agents = {}
        self.on_sub_agent_event: typing.Optional[typing.Callable] = None
        self._tool_search_config = None
        self._tool_search_function_name: typing.Optional[str] = None
        if mcp_servers:
            # add_mcp_server is async; calling it from __init__ silently
            # created never-awaited coroutines and registered nothing.
            raise ValueError(
                "mcp_servers cannot be registered from the constructor because "
                "registration is async. Use `await agent.add_mcp_server(server)` "
                "instead."
            )

    def _check_cancel(self):
        """Raise StopLoopException if the cancel event has been set."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise StopLoopException("Cancelled via cancel_event")

    def _is_tool_result_message(self, message: typing.Optional[Message]) -> bool:
        """True if the message carries tool/function results.

        Compaction must not run when the next message is a tool result, because
        wiping history would orphan the corresponding tool_use block in the
        prior assistant message and the API would reject the request.
        """
        if message is None:
            return False
        if message.role in ("function", "tool"):
            return True
        return any(
            part.type in ("function_result", "function_result_image")
            for part in (message.parts or [])
        )

    def _has_unanswered_tool_use(self) -> bool:
        """True if the last message is an assistant turn with pending tool_use.

        Compacting in this state would drop the tool_use blocks while a future
        tool_result still references their ids — the same orphan condition as
        above, just observed from the other side.
        """
        if not self.messages:
            return False
        last = self.messages[-1]
        if last.role != "assistant":
            return False
        return any(part.type == "function_call" for part in (last.parts or []))

    def _compaction_allowed(self, next_message: typing.Optional[Message]) -> bool:
        """True if compaction is configured and safe to run right now."""
        if not (self.compaction and hasattr(self.compaction, "should_compact")):
            return False
        if self._is_tool_result_message(next_message):
            return False
        if self._has_unanswered_tool_use():
            return False
        return True

    async def _maybe_compact(self, next_message: typing.Optional[Message]) -> bool:
        """Run compaction if safe. Returns True if compaction executed."""
        if not self._compaction_allowed(next_message):
            return False
        if not await self.compaction.should_compact(self):
            return False
        await self._ensure_tools_states(next_message)
        await self._encode_images_off_loop()
        await self.compaction.compact(self)
        return True

    @classmethod
    def from_template(cls, chat_template: str, **kwargs):
        instruction, examples = parse_chat_template(chat_template)
        return cls(
            instruction=instruction,
            examples=examples,
            **kwargs,
        )

    def reset(self):
        """Reset the chat state.

        This will clear the chat history and reset the functions and tools.
        """
        self.messages = []
        self.functions_schema = []
        self.functions = {}
        self.tools = {}
        self._sub_agents = {}
        self.on_sub_agent_event = None
        self._initial_tools_states = None
        self._tools_states_pinned = False
        self._tools_states_taken_for = None
        self._tools_states_from_parent_turn = False
        self._tool_search_config = None
        self._tool_search_function_name = None

    def refresh_tools_states(self):
        """Clear and re-capture tool states.

        Useful when tools have mutated and you want to pick up the new state
        without resetting the entire conversation.  Also lifts a
        pin_tools_states() pin.
        """
        self._tools_states_pinned = False
        self._tools_states_from_parent_turn = False
        self._initial_tools_states = None
        self._initial_tools_states = self.initial_tools_states

    def pin_tools_states(self, states: str):
        """Pin the tool-states system block to a caller-provided snapshot.

        A new user message normally takes a fresh snapshot of the live
        ``__llm__()`` values (see _capture_tools_states), which rewrites the
        system prompt — and thus invalidates the whole cached prefix — as
        soon as any of them drifted.  A caller that persists the block across
        turns (e.g. per conversation, alongside its history) can pin it back
        here before running: the block then stays byte-identical from turn to
        turn and the cache survives.  Pass the exact string a previous
        ``initial_tools_states`` returned.  The pin holds until
        refresh_tools_states() or reset() lifts it.
        """
        self._initial_tools_states = states
        self._tools_states_pinned = True

    @property
    def last_message(self):
        if not self.messages:
            return None
        return self.messages[-1].content

    def _capture_tools_states(self, message: typing.Optional[Message] = None):
        """Freeze the tool states used by the system prompt for this turn.

        Tool ``__llm__()`` values are live (row counts, background-filled
        hints, today's date, ...), so recomputing them between the iterations
        of one turn rewrites the system prompt and invalidates the whole
        cached prefix mid-loop.  Capture once and reuse: a new human message
        starts a new turn and takes a fresh snapshot, tool results (role
        "function") keep the current one.  reset() / refresh_tools_states()
        clear it explicitly; a pin_tools_states() pin blocks the new-turn
        recapture entirely.
        """
        if (
            message is not None
            and getattr(message, "role", None) == "user"
            and not self._tools_states_pinned
        ):
            self._initial_tools_states = None
        if self._initial_tools_states is None:
            self._initial_tools_states = self.initial_tools_states

    async def _capture_tools_states_async(
        self, message: typing.Optional[Message] = None
    ):
        """_capture_tools_states() for the async loop: same rule, off the loop.

        ``__llm__()`` is host code and routinely does I/O (a row count is a
        database round trip).  Called inline it would freeze the event loop,
        and every other request served by it, until the tools have answered.
        """
        taken_ahead = message is not None and message is self._tools_states_taken_for
        self._tools_states_taken_for = None
        new_turn = message is not None and getattr(message, "role", None) == "user"
        if (
            new_turn
            and not self._tools_states_pinned
            and not taken_ahead
            and not self._tools_states_from_parent_turn
        ):
            self._initial_tools_states = None
        if self._initial_tools_states is None:
            self._initial_tools_states = await self._compute_tools_states_off_loop()
        # An invocation of a sub-agent captured with its parent's turn starts
        # a "turn" of its own: its sub-agents were captured then too.
        if new_turn and not self._tools_states_from_parent_turn:
            await self._capture_sub_agent_states_off_loop()

    def _sub_agents_captured_at_turn_start(self) -> list:
        """The sub-agent templates whose states this agent's turn captures:
        registered with capture_states_at_turn_start, not pinned by the host,
        and theirs in turn, depth first."""
        templates = []
        for entry in self._sub_agents.values():
            agent = entry["agent"]
            if not entry.get("capture_states_at_turn_start"):
                continue
            if not agent._tools_states_pinned:
                templates.append(agent)
            templates.extend(agent._sub_agents_captured_at_turn_start())
        return templates

    async def _capture_sub_agent_states_off_loop(self):
        """Capture, at the start of this agent's turn, the states of the
        sub-agents registered with capture_states_at_turn_start.

        Left to their invocations, they would be captured in the middle of a
        tool batch, while the batch's other tools run on host state the
        sub-agent's tools often share (one database session).  Here nothing
        else of the turn runs yet.  Same thread and lock as any capture, one
        sub-agent after the other.  A sub-agent whose states fail is left to
        the capture at its invocation, where the failure surfaced before: it
        does not fail a turn that may never call it.
        """
        templates = self._sub_agents_captured_at_turn_start()
        if not templates:
            return
        lock = _tools_states_lock()
        # Read on the event loop, like _compute_tools_states_off_loop.
        jobs = [(agent, agent._tools_with_states()) for agent in templates]

        def render():
            results = []
            with lock:
                for agent, tools in jobs:
                    if not agent.tools:
                        results.append((agent, None, None))
                        continue
                    try:
                        results.append((agent, agent._render_tools_states(tools), None))
                    except Exception as exc:  # host code: anything can fail
                        results.append((agent, None, exc))
            return results

        for agent, states, error in await _run_in_thread(render):
            if error is not None:
                logging.warning(
                    "Tool states of sub-agent %r failed at the start of the "
                    "turn; they will be captured at its invocation",
                    agent.name,
                    exc_info=error,
                )
            agent._initial_tools_states = states
            agent._tools_states_from_parent_turn = error is None and states is not None

    async def _ensure_tools_states(self, next_message: typing.Optional[Message]):
        """Give a request sent ahead of the turn's capture a snapshot to read.

        Compaction runs before the capture, and its summary request carries
        the system prompt.  An agent with a snapshot keeps it for that request
        (the previous turn's, which is what the prompt cache holds).  An agent
        without one — history just loaded from storage — would have the
        provider compute the states inline, on the event loop.  Take the turn's
        snapshot now instead; the capture that follows keeps it rather than
        calling every ``__llm__()`` a second time.
        """
        if self._initial_tools_states is None:
            self._initial_tools_states = await self._compute_tools_states_off_loop()
            self._tools_states_taken_for = next_message

    async def _compute_tools_states_off_loop(self) -> typing.Optional[str]:
        """The tool states, with every ``__llm__()`` called on a worker thread.

        One thread for the whole capture, the tools one after the other: host
        tools commonly share one database session, which must not be used by
        two threads at once.  For the same reason captures started from the
        same event loop take turns, waiting on the worker thread, never on
        the loop.
        """
        if not self.tools:
            return None
        lock = _tools_states_lock()
        # Read here, on the event loop: self.tools can change while the worker
        # runs (an async tool registering another tool, for one).
        tools = self._tools_with_states()

        def render():
            with lock:
                return self._render_tools_states(tools)

        return await _run_in_thread(render)

    async def _encode_images_off_loop(self):
        """Encode the conversation's images on a worker thread, ahead of a request.

        The provider serializes every image part while it builds the request,
        on the event loop.  With PIL that is a full decode and re-encode of
        each image not serialized before, during which the loop serves nobody.
        Encoded here first, the request builder only reads what
        ``Image.to_base64()`` memoized.
        """
        pending = [
            part.image
            for message in self.messages
            for part in message.parts
            if isinstance(part.image, Image) and part.image._base64_cache is None
        ]
        if not pending:
            return

        def encode():
            for image in pending:
                with _image_encoding_lock:
                    try:
                        image.to_base64()
                    except ValueError:
                        # Left to the request builder, which raises it where
                        # it always has: inside the provider call.
                        pass

        await _run_in_thread(encode)

    @property
    def initial_tools_states(self) -> typing.Optional[str]:
        """Captured tool states included in the system prompt.

        Returns the snapshot taken for the current turn if there is one,
        otherwise computes from the current tools.  The snapshot is taken by
        _capture_tools_states() on each new user message and stays static for
        every iteration of that turn.

        Computed on the calling thread.  On the thread of a running event
        loop, it takes its turn among that loop's captures: one in progress on
        a worker thread is waited for, since nothing else ran on that thread
        during a capture when captures ran there.
        """
        if self._initial_tools_states is not None:
            return self._initial_tools_states
        # If there are no tools, return None
        if not self.tools:
            return None
        tools = self._tools_with_states()
        try:
            lock = _tools_states_lock()
        except RuntimeError:
            # No event loop running on this thread: nothing to take turns with.
            return self._render_tools_states(tools)
        with lock:
            return self._render_tools_states(tools)

    def _tools_with_states(self) -> list:
        """(name, tool) for each tool whose state goes in the system prompt."""
        # When tool search is enabled, skip __llm__ for tools whose
        # methods are ALL deferred — they haven't been discovered yet.
        loaded_tool_names = None
        if self._tool_search_config:
            loaded_tool_names = {
                s["name"].split("__", 1)[0]
                for s in self.functions_schema
                if not s.get("defer_loading")
            }

        tools = []
        for tool_id, tool in self.tools.items():
            tool_name = f"{tool.__class__.__name__}-{tool_id}"

            if loaded_tool_names is not None and tool_name not in loaded_tool_names:
                continue
            tools.append((tool_name, tool))
        return tools

    @staticmethod
    def _render_tools_states(tools: list) -> str:
        """Call ``__llm__()`` on each tool and assemble the system-prompt block."""
        tool_reprs = []
        for tool_name, tool in tools:
            if hasattr(tool, "__llm__"):
                tool_output = tool.__llm__()
            else:
                tool_output = (tool.__class__.__doc__ or "").strip()
            tool_output = truncate_with_marker(
                tool_output,
                OUTPUT_SIZE_LIMIT,
                label=tool_name,
                advice=" This tool's state is too large to show in full.",
            )
            tool_reprs.append(f"### {tool_name}\n{tool_output}")
        tool_context = "\n".join(tool_reprs)

        """
        ## Initial Tools States
        ### Tool 1
        {state}
        ### Tool 2
        {state}
        --- End of Initial Tools States ---
        """
        return f"## Initial Tools States\n{tool_context}\n--- End of Initial Tools States ---"

    @property
    def tool_search_categories_hint(self) -> typing.Optional[str]:
        """Short hint listing deferred tool categories for the system prompt.

        Only generated when tool search is enabled.  Helps the LLM know
        what kinds of tools it can search for without seeing their schemas.
        """
        if not self._tool_search_config:
            return None

        deferred = [s for s in self.functions_schema if s.get("defer_loading")]
        if not deferred:
            return None

        # Group tools by class/prefix (e.g. "EchartsTool", "CatalogTool")
        categories: dict[str, list[str]] = {}
        for s in deferred:
            name = s["name"]
            # Extract category from "ClassName-id__method" or use the name
            if "__" in name:
                prefix = name.split("__")[0]
                # Strip the tool_id suffix: "ClassName-id" -> "ClassName"
                if "-" in prefix:
                    category = prefix.rsplit("-", 1)[0]
                else:
                    category = prefix
            else:
                category = name
            method = name.split("__")[-1] if "__" in name else name
            categories.setdefault(category, []).append(method)

        lines = []
        for category, methods in categories.items():
            methods_str = ", ".join(methods)
            lines.append(f"- {category}: {methods_str}")

        return (
            "## Searchable Tool Categories\n"
            "Use the tool_search function to discover and load these tools:\n"
            + "\n".join(lines)
        )

    def load_messages(self, messages: list[Message], keep_thinking: bool = False):
        """Restore a conversation from the caller's storage.

        ``keep_thinking`` replays the stored thinking blocks instead of
        dropping them.  Anthropic's cache is a prefix match, so an assistant
        message that is serialized without its thinking blocks after having
        been *sent* with them changes the prefix at that point: every new
        question then rewrites the whole conversation instead of reading it
        from cache.  Regular thinking blocks are not origin-locked — the
        server renders them into the target model's prompt, so replaying one
        produced by another Claude model is safe — but a block whose
        signature was lost in storage is rejected, so those are still
        dropped.  Opt in only if you persist ``thinking`` and
        ``thinking_signature`` byte-for-byte, in their original order.
        """
        # If compaction checkpoints exist, only load from the latest one onward
        # (older messages were already summarized into the compaction message)
        latest_compaction_idx = None
        for i, msg in enumerate(messages):
            if any(p.type == "compaction" for p in (msg.parts or [])):
                latest_compaction_idx = i

        if latest_compaction_idx is not None:
            messages = messages[latest_compaction_idx:]

        for msg in messages:
            if msg.role != "assistant":
                continue
            if not keep_thinking:
                # Default: drop them.  A caller that re-serializes thinking
                # from storage may have lost the signature or the ordering,
                # and the Anthropic docs confirm omitting thinking from prior
                # turns is accepted.
                msg.parts = [p for p in msg.parts if p.type != "thinking"]
                continue
            # A thinking block needs its signature — send one without and the
            # API rejects the request.
            msg.parts = [
                p for p in msg.parts if p.type != "thinking" or p.thinking_signature
            ]
            # Tell the provider these blocks are replayable as-is.
            msg.is_live = any(p.type == "thinking" for p in msg.parts)

        # Remove messages that became empty after thinking block removal
        # (e.g. assistant messages that contained only thinking parts)
        messages = [m for m in messages if m.parts]

        self._rehydrate_tool_references(messages)

        self.messages = messages

    def _rehydrate_tool_references(self, messages: list[Message]) -> None:
        """Rebuild ``tool_references`` on stored tool-search results.

        A discovered tool stays loaded only through the ``tool_reference``
        blocks the provider emits from ``MessagePart.tool_references``.
        Callers usually persist the part's text (``"name_a, name_b"``), not
        the list, so a reloaded history would silently drop every tool the
        model had found and force a new search on the next turn.  Parse the
        text back when this agent has tool search.

        Existing references are filtered on every agent, tool search or not:
        a history replayed under another tool set (a narrower agent, a tool
        renamed or removed since) would otherwise name tools missing from the
        request, and Anthropic rejects it with 400 "Tool reference '…' not
        found in available tools".  Register tools before loading messages.
        """
        search_name = self._tool_search_function_name
        for msg in messages:
            if msg.role != "function":
                continue
            for part in msg.parts:
                if part.type != "function_result":
                    continue
                if part.tool_references is not None:
                    part.tool_references = [
                        name for name in part.tool_references if name in self.functions
                    ]
                elif search_name is not None and msg.name == search_name:
                    part.tool_references = self._parse_tool_references(part.content)

    def _parse_tool_references(self, content: typing.Optional[str]) -> list[str]:
        if not content or content.strip() == "[]":
            return []
        names = [name.strip() for name in content.split(",")]
        return [name for name in names if name and name in self.functions]

    def add_function(
        self,
        function: typing.Callable,
        function_schema: typing.Optional[dict] = None,
        defer_loading: bool = False,
    ):
        if function_schema is None:
            # We try to infer the function schema from the function
            function_schema = inspect_schema(function)

        if defer_loading:
            function_schema["defer_loading"] = True

        # Auto-defer when tool search is enabled
        if (
            self._tool_search_config
            and not defer_loading
            and function_schema["name"] not in self._tool_search_config.always_loaded
            and function_schema["name"] != self._tool_search_function_name
        ):
            function_schema["defer_loading"] = True

        # Re-registering a name replaces the previous function. Dispatch
        # (self.functions) is last-wins, so the schema list must follow:
        # a duplicate name in functions_schema reaches the provider payload
        # and Anthropic/OpenAI reject the whole request ("tool names must
        # be unique").
        name = function_schema["name"]
        existing_index = next(
            (i for i, s in enumerate(self.functions_schema) if s["name"] == name),
            None,
        )
        if existing_index is not None:
            self.functions_schema[existing_index] = function_schema
        else:
            self.functions_schema.append(function_schema)
        self.functions[name] = function

    def add_tool(
        self, tool: typing.Union[type, object], tool_id: typing.Optional[str] = None
    ) -> str:
        """Add a tool class or instance to the chat instance and return the tool id.

        Re-registering an existing tool_id replaces the previous tool: its
        registrations are removed first, so methods that only existed on the
        old tool (e.g. when the class differs) don't linger.
        """
        if isinstance(tool, type):
            # If a class is provided, instantiate it
            tool = tool()

        if tool_id is None:
            tool_id = str(id(tool))

        if tool_id in self.tools:
            self.remove_tool(tool_id)
        self.tools[tool_id] = tool

        class_name = tool.__class__.__name__
        tool_name = f"{class_name}-{tool_id}"
        # Get all methods and functions from the tool
        functions = inspect.getmembers(tool, inspect.ismethod) + inspect.getmembers(
            tool, inspect.isfunction
        )
        for function_name, function in functions:
            if function_name.startswith("_"):  # Skip private methods
                continue
            function_schema = inspect_schema(function)
            function_schema["name"] = f"{tool_name}__{function_name}"
            self.add_function(function, function_schema)
        return tool_id

    def remove_tool(self, tool_id: str):
        tool = self.tools.pop(tool_id)
        prefix = f"{tool.__class__.__name__}-{tool_id}__"
        self.functions_schema = [
            schema
            for schema in self.functions_schema
            if not schema["name"].startswith(prefix)
        ]
        self.functions = {
            name: func
            for name, func in self.functions.items()
            if not name.startswith(prefix)
        }

    def add_sub_agent(
        self,
        agent: "Agentlys",
        name: typing.Optional[str] = None,
        description: typing.Optional[str] = None,
        compute_levels: typing.Optional[typing.Union[bool, dict]] = None,
        capture_states_at_turn_start: bool = False,
    ) -> str:
        """Add a sub-agent that can be triggered by this agent.

        The sub-agent runs its own conversation loop when called and only
        returns its final response to this agent's context, keeping the
        main context window clean.

        Args:
            agent: An Agentlys instance to use as a sub-agent.
            name: Override name for the sub-agent tool. Defaults to agent.name.
            description: Description shown to the LLM. Defaults to agent.instruction.
            compute_levels: Enable dynamic compute levels for this sub-agent.
                - None/False: disabled (default, backward compatible).
                - True: enabled with default Anthropic mapping (opus/sonnet/haiku).
                - dict: custom mapping with keys "high", "medium", "low" mapped to model names.
            capture_states_at_turn_start: Capture the sub-agent's tool states
                when this agent's async turn starts, not at each invocation.
                Its tools' ``__llm__()`` then never run in the middle of a tool
                batch, where they could share host state (a database session)
                with the batch's other tools, and every invocation of the turn
                sends the same block.  They run at every turn, invoked or not.
                A pin_tools_states() pin on the sub-agent still wins.

        Returns:
            The registered function name (e.g. "sub_agent__researcher").
        """
        sub_agent_name = name or agent.name
        if not sub_agent_name:
            raise ValueError(
                "Sub-agent must have a name. Set agent.name or pass name= to add_sub_agent()."
            )

        func_name = f"sub_agent__{sub_agent_name}"

        if func_name in self._sub_agents:
            raise ValueError(
                f"Sub-agent with name '{sub_agent_name}' is already registered."
            )

        # Resolve compute level mapping
        compute_mapping = None
        if compute_levels is True:
            compute_mapping = DEFAULT_COMPUTE_LEVELS.copy()
        elif isinstance(compute_levels, dict):
            required_keys = {"high", "medium", "low"}
            missing = required_keys - set(compute_levels.keys())
            if missing:
                raise ValueError(
                    f"compute_levels dict is missing required keys: {missing}"
                )
            compute_mapping = compute_levels

        async def _invoke_sub_agent(prompt: str, compute_level: str = "medium") -> str:
            """Run the sub-agent with the given prompt and return its response."""
            invocation_id = str(uuid.uuid4())

            # Always create a shallow copy to isolate state — parallel calls
            # to the same sub-agent must not race on shared messages/provider.
            run_agent = copy.copy(agent)
            run_agent.provider = copy.copy(agent.provider)
            run_agent.provider.chat = run_agent
            if (
                self.cancel_event is not None
            ):  # Propagate cancellation, preserve child's own event
                run_agent.cancel_event = self.cancel_event
            run_agent.messages = []

            if compute_mapping:
                if compute_level not in compute_mapping:
                    logging.warning(
                        f"Invalid compute_level '{compute_level}', falling back to 'medium'"
                    )
                    compute_level = "medium"
                target_model = compute_mapping[compute_level]
                run_agent.model = target_model
                run_agent.provider.model = target_model
                run_agent.thinking = _resolve_thinking_for_model(
                    agent.thinking, target_model
                )

            final_content = None
            async for event in run_agent.run_conversation_stream_async(prompt):
                # Forward events to parent via callback
                if self.on_sub_agent_event:
                    self.on_sub_agent_event(sub_agent_name, invocation_id, event)

                # Capture final assistant response
                if event.get("type") == "assistant":
                    msg = event.get("message")
                    if msg and msg.content:
                        final_content = msg.content

            return final_content or "Sub-agent produced no response."

        func_description = (
            description
            or agent.instruction
            or f"Delegate a task to the {sub_agent_name} sub-agent."
        )

        properties = {
            "prompt": {
                "type": "string",
                "description": "The task or question to send to this sub-agent.",
            }
        }
        if compute_mapping:
            properties["compute_level"] = {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "Compute level: 'high' for complex tasks requiring deep reasoning, "
                    "'medium' for standard tasks, 'low' for simple lookups. "
                    "Defaults to 'medium'."
                ),
            }

        function_schema = {
            "name": func_name,
            "description": func_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["prompt"],
            },
        }

        self.add_function(_invoke_sub_agent, function_schema)
        self._sub_agents[func_name] = {
            "agent": agent,
            "compute_levels": compute_mapping,
            "capture_states_at_turn_start": capture_states_at_turn_start,
        }

        return func_name

    @property
    def sub_agents(self) -> dict:
        """The registered sub-agents, by function name (``sub_agent__<name>``).

        Each value is the template agent: every invocation runs on a copy of
        it, which starts from its tool states, so a host can pin them here
        (pin_tools_states) before a run.  A new dict at each read: register
        and remove sub-agents with add_sub_agent / remove_sub_agent.
        """
        return {
            func_name: entry["agent"] for func_name, entry in self._sub_agents.items()
        }

    def remove_sub_agent(self, name: str):
        """Remove a sub-agent by name.

        Args:
            name: The sub-agent name (without the 'sub_agent__' prefix).
        """
        func_name = f"sub_agent__{name}"
        if func_name not in self._sub_agents:
            raise ValueError(f"No sub-agent with name '{name}' found.")

        del self._sub_agents[func_name]
        self.functions.pop(func_name, None)
        self.functions_schema = [
            schema for schema in self.functions_schema if schema["name"] != func_name
        ]

    def enable_tool_search(
        self,
        always_loaded: typing.Optional[list[str]] = None,
        search_fn: typing.Optional[typing.Callable] = None,
        search_model: typing.Optional[str] = None,
    ):
        """Enable tool search to defer most tools and discover them on-demand.

        When enabled, all registered tools (except those in *always_loaded*)
        are marked with ``defer_loading=True`` so the provider hides them
        from the LLM's context.  A search tool is automatically registered
        that the LLM can call to discover relevant tools.

        Tools added *after* this call are also auto-deferred unless they
        appear in *always_loaded*.

        Args:
            always_loaded: Tool names to keep always visible to the LLM.
                If *None*, no tools are always-loaded (besides the search
                tool itself).
            search_fn: Custom async search function with signature
                ``async def search(query: str, catalog: list[dict]) -> list[str]``.
                Each catalog entry has ``name`` and ``description`` keys.
                If *None*, uses a default LLM-based search with a cheap model.
            search_model: Model to use for the default LLM-based search.
                Ignored if *search_fn* is provided.  Defaults to
                ``claude-haiku-4-5-20251001`` for Anthropic providers and
                ``gpt-4o-mini`` for OpenAI providers.
        """
        from agentlys.tool_search import ToolSearchConfig, create_search_tool_fn

        if always_loaded is None:
            always_loaded = []

        # Determine default search model based on provider
        if search_model is None:
            provider_name = self.provider.__class__.__name__.lower()
            if "openai" in provider_name:
                search_model = "gpt-4o-mini"
            else:
                search_model = "claude-haiku-4-5-20251001"

        # Remove previous search tool if re-enabling
        if self._tool_search_function_name is not None:
            self.functions.pop(self._tool_search_function_name, None)
            self.functions_schema = [
                s
                for s in self.functions_schema
                if s["name"] != self._tool_search_function_name
            ]

        self._tool_search_config = ToolSearchConfig(
            always_loaded=always_loaded,
            search_fn=search_fn,
            search_model=search_model,
        )

        # Mark all existing functions as deferred (except always_loaded).
        # Also clear defer_loading for tools that are now always_loaded
        # (handles reconfiguration with a different always_loaded list).
        for schema in self.functions_schema:
            if schema["name"] in always_loaded:
                schema.pop("defer_loading", None)
            else:
                schema["defer_loading"] = True

        # Register the search tool (never deferred)
        search_callable, search_schema = create_search_tool_fn(self)
        self._tool_search_function_name = search_schema["name"]
        self.add_function(search_callable, search_schema)

    async def add_mcp_server(
        self,
        mcp_server,
        prefix: str = "",
        tool_filter: typing.Optional[typing.Callable] = None,
        max_result_chars: typing.Optional[int] = None,
        include_resources: bool = True,
    ) -> list[str]:
        """
        Add a MCP server to the agent instance.
        The MCP server will be used to call tools and resources.

        Args:
            mcp_server: An initialized MCP ``ClientSession`` (or any object
                exposing the same ``list_tools`` / ``call_tool`` /
                ``list_resources`` interface).
            prefix: Namespace prefix applied to every registered name so
                tools from different servers (or built-in tools) never
                collide.
            tool_filter: Optional callable receiving the MCP ``Tool``
                object; return False to skip the tool. Called on the event
                loop for each tool: it must not block.
            max_result_chars: Optional cap on tool/resource result size.
            include_resources: Also expose the server's resources as
                functions (default True, matching historical behavior).

        Returns:
            The list of registered function names. Names already taken by
            existing functions are skipped (with a warning), so MCP servers
            can never shadow built-in tools.
        """
        from agentlys.mcp import fetch_mcp_server_resources, fetch_mcp_server_tools

        registered: list[str] = []

        batches = [
            await fetch_mcp_server_tools(
                mcp_server,
                prefix=prefix,
                tool_filter=tool_filter,
                max_result_chars=max_result_chars,
            )
        ]
        if include_resources:
            batches.append(
                await fetch_mcp_server_resources(
                    mcp_server,
                    prefix=prefix,
                    max_result_chars=max_result_chars,
                )
            )

        # Route through add_function so tool-search auto-deferral applies
        # and the registration behaves like any other tool.
        for functions, schemas in batches:
            for schema in schemas:
                if schema["name"] in self.functions:
                    logging.warning(
                        "MCP name %r collides with an existing function, skipping",
                        schema["name"],
                    )
                    continue
                self.add_function(functions[schema["name"]], schema)
                registered.append(schema["name"])
        return registered

    async def ask_async(
        self,
        message: typing.Union[Message, str, None] = None,
        **kwargs,
    ) -> Message:
        """Async version of ask method that uses the async fetch_async provider method"""
        if message:
            if isinstance(message, str):
                # If message is instance of string, then convert to Message
                message = Message(
                    role="user",
                    content=message,
                )

        # Run compaction before appending the new message so it is never
        # included in the summary — the LLM must see the exact user request.
        # Skipped mid tool exchange to avoid orphaning tool_use/tool_result.
        await self._maybe_compact(message)

        if message:
            self.messages.append(message)

        await self._capture_tools_states_async(message)
        await self._encode_images_off_loop()

        # Merge class-level thinking with any kwargs override
        if self.thinking and "thinking" not in kwargs:
            kwargs["thinking"] = self.thinking

        # Call the async strategy
        response = await self.provider.fetch_async(**kwargs)
        self.messages.append(response)
        return response

    def ask(
        self,
        message: typing.Union[Message, str, None] = None,
        **kwargs,
    ) -> Message:
        # For backward compatibility, use run_until_complete to execute the async method
        loop = get_event_loop_or_create()
        return loop.run_until_complete(self.ask_async(message, **kwargs))

    def _format_callback_message(self, function_name, function_call_id, content, image):
        if isinstance(content, Message):
            message = content
            # TODO: Add name and function_call_id to the message
            # message.name = function_name
            # message.function_call_id = function_call_id
            return message

        # Tool search results get special handling — the provider layer
        # will convert tool_references into provider-specific blocks
        # (e.g. Anthropic tool_reference content blocks).
        if (
            self._tool_search_function_name
            and function_name == self._tool_search_function_name
            and isinstance(content, list)
            and all(isinstance(item, str) for item in content)
        ):
            part = MessagePart(
                type="function_result",
                content=", ".join(content) if content else "[]",
                function_call_id=function_call_id,
                tool_references=content,
            )
            return Message(name=function_name, role="function", parts=[part])

        # We format the function_call response to be used as a next message
        if content is None:
            # When content is None, use an empty string instead to prevent the "Message should have at least one part" error
            formatted_content = ""
        elif isinstance(content, list):
            formatted_content = []
            if not content:
                formatted_content = "[]"
            elif isinstance(content[0], dict):
                # If data is list of dicts, dumps to CSV
                formatted_content = csv_dumps(content, OUTPUT_SIZE_LIMIT)
            else:
                for item in content:
                    if isinstance(item, str):
                        formatted_content.append(item)
                    elif isinstance(item, (bool, int, float)):
                        formatted_content.append(str(item))
                    else:
                        tool_id = self.add_tool(item)
                        formatted_content.append(f"Added tool: {tool_id}")
                formatted_content = "\n".join(formatted_content)
                # Limit the size of the content
                formatted_content = truncate_with_marker(
                    formatted_content, OUTPUT_SIZE_LIMIT, label=function_name
                )
        elif isinstance(content, _SerializedDict):
            formatted_content = truncate_with_marker(
                content.head,
                OUTPUT_SIZE_LIMIT,
                label=function_name,
                full_length=content.length,
            )
        elif isinstance(content, dict):
            # default=str: tool results routinely carry values json doesn't
            # know (Decimal, datetime, UUID, ...); stringify instead of crashing
            content_dump = json.dumps(content, default=str)
            formatted_content = truncate_with_marker(
                content_dump, OUTPUT_SIZE_LIMIT, label=function_name
            )
        elif isinstance(content, str):
            formatted_content = truncate_with_marker(
                content, OUTPUT_SIZE_LIMIT, label=function_name
            )
        elif isinstance(content, bytes):
            # Detect if it's an image
            try:
                image = PILImage.open(io.BytesIO(content))
                formatted_content = None
            except IOError:
                # Not an image
                raise ValueError("Returned bytes is not a valid image.")
        elif isinstance(content, PILImage.Image):
            image = content
            formatted_content = None
        elif isinstance(content, (int, float, bool)):
            formatted_content = str(content)
        elif isinstance(content, object):
            # If the function return an object, we add it as a tool
            # NOTE: Maybe we shouldn't, and rely on a clearer signal / object type ?
            tool_id = self.add_tool(content)
            formatted_content = f"Added tool: {tool_id}"
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

        # Build a function response message.
        # If there's an image, create a single function_result_image part that includes
        # both the text content and the image. This ensures we have exactly ONE tool_result
        # per tool_use_id.
        if image:
            part = MessagePart(
                type="function_result_image",
                content=formatted_content,  # Include text content in the image part
                image=image,
                function_call_id=function_call_id,
            )
        else:
            part = MessagePart(
                type="function_result",
                content=formatted_content,
                function_call_id=function_call_id,
            )

        return Message(name=function_name, role="function", parts=[part])

    def _format_callback_message_safe(
        self, function_name, function_call_id, content, image
    ):
        """Format a tool result under the same error boundary as tool execution.

        Formatting runs after _execute_single_tool's try/except, so a return
        value that can't be formatted must become an error function_result the
        model can react to — not an exception that kills the whole turn.
        """
        try:
            return self._format_callback_message(
                function_name=function_name,
                function_call_id=function_call_id,
                content=content,
                image=image,
            )
        except StopLoopException:
            raise
        except Exception as e:
            return Message(
                name=function_name,
                role="function",
                parts=[
                    MessagePart(
                        type="function_result",
                        content=self._format_exception(e),
                        function_call_id=function_call_id,
                    )
                ],
            )

    def _format_exception(self, e):
        # We clean the traceback to remove frames from __init__.py, plus the
        # asyncio.to_thread plumbing frames that appear when a sync tool is
        # offloaded to a worker thread — those add no value for the LLM.
        tb = traceback.extract_tb(e.__traceback__)
        agentlys_internal_files = (
            "chat.py",
            "asyncio/threads.py",
            "concurrent/futures/thread.py",
        )
        filtered_tb = [
            frame
            for frame in tb
            if not any(marker in frame.filename for marker in agentlys_internal_files)
        ]
        if filtered_tb:
            content = "Traceback (most recent call last):\n"
            content += "".join(traceback.format_list(filtered_tb))
            content += f"\n{e.__class__.__name__}: {str(e)}"
        else:
            # If no relevant frames, use the full traceback
            content = traceback.format_exc()
        return content

    async def _call_with_signature(self, func, from_response, **kwargs):
        # Per-call isolation seam. A tool method runs concurrently with others
        # in a turn — sync tools on worker threads (see below), async tools as
        # interleaved tasks — yet they share the one tool instance. State on
        # that instance which isn't safe to share (a DB session, an open
        # transaction, a non-thread-safe client) then races across calls.
        #
        # If the tool owning this method exposes ``__agentlys_call_scope__`` (a
        # context manager), run the call inside it and rebind the method to the
        # instance it yields, so each concurrent invocation gets its own scope.
        # Tools without it are unaffected; plain functions have no owner.
        owner = getattr(func, "__self__", None)
        call_scope = getattr(owner, "__agentlys_call_scope__", None)
        if call_scope is None:
            return await self._invoke_func(func, from_response, **kwargs)

        # Async tools run as tasks on the event loop, so opening the scope here
        # and awaiting the body share one thread/task — enter the scope inline.
        # A scope that blocks therefore blocks the loop: a tool whose scope
        # does I/O should be sync.
        if self._is_async_callable(func):
            with call_scope() as scoped_owner:
                scoped_func = getattr(scoped_owner, func.__name__)
                return await self._invoke_func(scoped_func, from_response, **kwargs)

        # Sync tools are offloaded to a worker thread (see ``_invoke_func``), so
        # the scope must open, run AND close on that same thread. Entering it on
        # the event-loop thread would create/close a thread-affine resource (a
        # SQLAlchemy ``Session``) on a different thread than the one the method
        # body uses, defeating the "session per thread" isolation this provides.
        def _run_scoped_sync():
            with call_scope() as scoped_owner:
                scoped_func = getattr(scoped_owner, func.__name__)
                call_kwargs = kwargs
                if _accepts_from_response(scoped_func):
                    call_kwargs = {**kwargs, "from_response": from_response}
                return scoped_func(**call_kwargs)

        result = await asyncio.to_thread(_run_scoped_sync)
        # Defensive: a sync wrapper may return a coroutine (eg a lambda around an
        # async fn); await it on the event loop to preserve historical semantics.
        if inspect.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _is_async_callable(func) -> bool:
        return inspect.iscoroutinefunction(func) or (
            callable(func)
            and inspect.iscoroutinefunction(getattr(func, "__call__", None))
        )

    async def _invoke_func(self, func, from_response, **kwargs):
        if _accepts_from_response(func):
            kwargs = {**kwargs, "from_response": from_response}

        # Async tools: await them directly on the event loop.
        if self._is_async_callable(func):
            return await func(**kwargs)

        # Sync tools: offload to a worker thread so a slow call (network I/O,
        # blocking DB driver, …) can never freeze the asyncio event loop and
        # stall every other in-flight request behind it.
        result = await asyncio.to_thread(func, **kwargs)

        # Defensive: some sync wrappers return a coroutine (eg lambdas around
        # async fns). Await it to preserve historical semantics.
        if inspect.iscoroutine(result):
            return await result
        return result

    async def _resolve_and_call_function(
        self, name: str, args: dict, response: Message
    ) -> typing.Any:
        """Resolve function/tool by name and call it with args.

        Tool methods are registered in self.functions like any other
        function (see add_tool), so a single lookup covers both.
        """
        return await self._call_with_signature(self.functions[name], response, **args)

    def _process_tool_result(self, result: typing.Any) -> tuple[typing.Any, typing.Any]:
        """Process tool result and handle tuple unpacking.

        Returns:
            Tuple of (content, image)
        """
        content = None
        image = None

        # Handle functions that return (content, image) tuples
        if isinstance(result, tuple) and len(result) == 2:
            content, image = result
        else:
            content = result

        return content, image

    async def _execute_single_tool(
        self,
        part: MessagePart,
        response: Message,
    ) -> tuple[str, str, typing.Any, typing.Any]:
        """Execute one tool and return (function_call_id, function_name, content, image)."""
        self._check_cancel()
        name = part.function_call["name"]
        args = part.function_call["arguments"]
        function_call_id = part.function_call_id

        if part.arguments_parsing_error is not None:
            raw = part.raw_arguments or ""
            preview = raw[:500] + ("…" if len(raw) > 500 else "")
            content = (
                f"Invalid JSON arguments for {name}: {part.arguments_parsing_error}."
                f"\nRaw arguments (first 500 characters): {preview}"
                "\nThe tool was not executed. Re-emit the tool call with valid JSON "
                "object arguments. Do not repeat other successful tool calls."
            )
            return (function_call_id, name, content, None)

        content = None
        image = None

        try:
            result = await self._resolve_and_call_function(name, args, response)
            content, image = self._process_tool_result(result)
            if isinstance(content, dict):
                content = await _serialize_dict(content)
        except StopLoopException:
            raise  # Propagate to stop the loop
        except Exception as e:
            # Also a result that cannot be serialized: the error replaces it
            # whole, its image included.
            content, image = self._format_exception(e), None

        return (function_call_id, name, content, image)

    async def _call_functions_parallel(
        self,
        function_call_parts: list[MessagePart],
        response: Message,
    ) -> Message:
        """
        Execute multiple tool calls in parallel and build a combined result message.

        Per Anthropic API requirements, all tool_results must be in a SINGLE user message.
        Uses asyncio.gather with return_exceptions=True to collect all results even if some fail.
        """
        # Execute all tools in parallel
        results = await asyncio.gather(
            *[
                self._execute_single_tool(part, response)
                for part in function_call_parts
            ],
            return_exceptions=True,  # Don't fail fast - collect all results
        )

        # Build combined message with all tool results
        parts = []
        formatted_messages = []
        for i, result in enumerate(results):
            if isinstance(result, StopLoopException):
                raise result
            # _execute_single_tool handles its own exceptions,
            # so this path is rare (e.g., task-level failures)
            if isinstance(result, Exception):
                function_call_id = function_call_parts[i].function_call_id
                function_name = function_call_parts[i].function_call["name"]
                error_content = self._format_exception(result)
                parts.append(
                    MessagePart(
                        type="function_result",
                        content=error_content,
                        function_call_id=function_call_id,
                    )
                )
                continue

            function_call_id, function_name, content, image = result
            formatted_msg = self._format_callback_message_safe(
                function_name=function_name,
                function_call_id=function_call_id,
                content=content,
                image=image,
            )
            formatted_messages.append(formatted_msg)
            # Extract parts from formatted message
            parts.extend(formatted_msg.parts)

        # For single function call, return the formatted message directly
        # to preserve the name attribute (important for OpenAI compatibility)
        if len(formatted_messages) == 1 and len(parts) == len(
            formatted_messages[0].parts
        ):
            return formatted_messages[0]

        # Create single message with all results for multiple tools
        return Message(role="function", parts=parts)

    async def _stream_functions_parallel(
        self,
        function_call_parts: list[MessagePart],
        response: Message,
    ) -> typing.AsyncGenerator[tuple[str, str, Message], None]:
        """
        Execute multiple tool calls in parallel, yielding results as each completes.

        Yields tuples of (function_call_id, function_name, result_message) as each tool finishes.
        This allows streaming individual results to the UI while tools execute in parallel.
        """
        # Create tasks - store as a list for asyncio.wait
        tasks = [
            asyncio.create_task(self._execute_single_tool(part, response))
            for part in function_call_parts
        ]

        # Use asyncio.wait with FIRST_COMPLETED to yield results as they finish
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    result = task.result()
                except StopLoopException:
                    # Cancel remaining tasks and await them before raising
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    raise
                except Exception as e:
                    # Handle unexpected exceptions - _execute_single_tool handles
                    # its own exceptions, so this path is rare
                    error_msg = Message(
                        role="function",
                        content=self._format_exception(e),
                        function_call_id=None,
                    )
                    yield (None, "unknown", error_msg)
                    continue

                function_call_id, function_name, content, image = result
                formatted_msg = self._format_callback_message_safe(
                    function_name=function_name,
                    function_call_id=function_call_id,
                    content=content,
                    image=image,
                )
                yield (function_call_id, function_name, formatted_msg)

    @staticmethod
    def _check_parsing_retry_budget(response: Message, retries: int) -> int:
        # Two re-asks per run; count failed turns, not parallel calls.
        invalid = [
            part
            for part in response.function_call_parts
            if part.arguments_parsing_error is not None
        ]
        if invalid:
            if retries >= 2:
                part = invalid[0]
                raise FunctionCallParsingError(
                    response.id,
                    {
                        "name": part.function_call["name"],
                        "arguments": part.raw_arguments,
                    },
                )
            return retries + 1
        return retries

    async def run_conversation_async(
        self,
        question: typing.Union[str, Message, None] = None,
    ) -> typing.AsyncGenerator[Message, None]:
        """Async version of run_conversation with parallel tool execution support."""
        # If there's an initial question, emit it:
        if isinstance(question, str):
            message = Message(
                role="user",
                content=question,
            )
            yield message
        else:
            message = question

        parsing_retries = 0
        for _ in range(self.max_interactions):
            self._check_cancel()
            # Ask the LLM with the current message (if any)
            response = await self.ask_async(message)

            # Get ALL function_call parts (supports parallel tool calls)
            function_call_parts = response.function_call_parts

            if not function_call_parts:
                yield response
                try:
                    message = self.simple_response_callback(response)
                except StopLoopException:
                    return
            else:
                yield response
                parsing_retries = self._check_parsing_retry_budget(
                    response, parsing_retries
                )

                try:
                    # Execute all tools in parallel
                    message = await self._call_functions_parallel(
                        function_call_parts, response
                    )
                except StopLoopException:
                    return

                yield message

    async def ask_stream_async(
        self,
        message: typing.Union[Message, str, None] = None,
        **kwargs,
    ) -> typing.AsyncGenerator[dict, None]:
        """Async streaming version of ask method.

        Yields:
            - {"type": "text", "content": str} - text chunks as they arrive
            - {"type": "message", "message": Message} - final complete message
        """
        if message:
            if isinstance(message, str):
                message = Message(role="user", content=message)

        # Run compaction before appending the new message so it is never
        # included in the summary — the LLM must see the exact user request.
        # Skipped mid tool exchange to avoid orphaning tool_use/tool_result.
        if self._compaction_allowed(message) and await self.compaction.should_compact(
            self
        ):
            yield {"type": "compacting"}
            first_message = self.messages[0]
            await self._ensure_tools_states(message)
            await self._encode_images_off_loop()
            await self.compaction.compact(self)
            # Yield the compaction summary message so callers can persist it.
            # Only emit when compact() actually produced a new summary: it can
            # be a no-op, leaving an earlier summary in first position.
            if (
                self.messages[0] is not first_message
                and self.messages[0].has_compaction
            ):
                yield {"type": "compaction_message", "message": self.messages[0]}

        if message:
            self.messages.append(message)

        await self._capture_tools_states_async(message)
        await self._encode_images_off_loop()

        final_message = None
        async for chunk in self.provider.fetch_stream_async(**kwargs):
            if chunk["type"] == "message":
                final_message = chunk["message"]
                self.messages.append(final_message)
            yield chunk

        if final_message is None:
            raise RuntimeError("Stream ended without final message")

    async def run_conversation_stream_async(
        self,
        question: typing.Union[str, Message, None] = None,
    ) -> typing.AsyncGenerator[dict, None]:
        """Run conversation with streaming text responses and parallel tool execution.

        Yields:
            - {"type": "user", "message": Message} - user message
            - {"type": "text", "content": str} - text chunks as they stream
            - {"type": "assistant", "message": Message} - complete assistant message
            - {"type": "tools_executing", "data": dict} - signal parallel tool execution start
            - {"type": "tool_result", "data": dict} - individual tool result as it completes
            - {"type": "function", "message": Message} - combined function results message
        """
        # Handle initial question
        if isinstance(question, str):
            message = Message(role="user", content=question)
            yield {"type": "user", "message": message}
        else:
            message = question

        parsing_retries = 0
        for _ in range(self.max_interactions):
            self._check_cancel()
            # Stream the LLM response
            response = None
            async for chunk in self.ask_stream_async(message):
                self._check_cancel()
                if chunk["type"] == "message":
                    response = chunk["message"]
                else:
                    # Forward all other events (text, tool_started, tool_delta, etc.)
                    yield chunk

            if response is None:
                raise RuntimeError("Stream ended without response")

            # Get ALL function_call parts (supports parallel tool calls)
            function_call_parts = response.function_call_parts

            if not function_call_parts:
                yield {"type": "assistant", "message": response}
                try:
                    message = self.simple_response_callback(response)
                except StopLoopException:
                    return
            else:
                # Yield assistant message with all tool calls
                yield {"type": "assistant", "message": response}

                parsing_retries = self._check_parsing_retry_budget(
                    response, parsing_retries
                )

                # Signal start of parallel execution
                yield {
                    "type": "tools_executing",
                    "data": {
                        "count": len(function_call_parts),
                        "tools": [
                            {
                                "name": p.function_call["name"],
                                "id": p.function_call_id,
                            }
                            for p in function_call_parts
                        ],
                    },
                }

                try:
                    # Execute tools in parallel, streaming results as they complete
                    all_parts = []
                    result_messages = []
                    async for (
                        function_call_id,
                        function_name,
                        result_msg,
                    ) in self._stream_functions_parallel(function_call_parts, response):
                        # Yield individual result for UI streaming
                        yield {
                            "type": "tool_result",
                            "data": {
                                "function_call_id": function_call_id,
                                "function_name": function_name,
                                "message": result_msg,
                            },
                        }
                        # Collect for combined message
                        result_messages.append(result_msg)
                        all_parts.extend(result_msg.parts)

                    # Build combined message for conversation history
                    # For single function call, use the formatted message directly
                    # to preserve the name attribute (important for OpenAI legacy)
                    if len(result_messages) == 1:
                        message = result_messages[0]
                    else:
                        message = Message(role="function", parts=all_parts)

                except StopLoopException:
                    return

                # Yield combined function results (for consumers that expect it)
                yield {"type": "function", "message": message}

    def run_conversation(self, question: typing.Union[str, Message, None] = None):
        # For backward compatibility, use run_until_complete to execute the async method
        loop = get_event_loop_or_create()
        async_gen = self.run_conversation_async(question)

        while True:
            try:
                # Get the next item from the async generator
                item = loop.run_until_complete(async_gen.__anext__())
                # Yield the item immediately
                yield item
            except StopAsyncIteration:
                # Stop when the async generator is exhausted
                break
