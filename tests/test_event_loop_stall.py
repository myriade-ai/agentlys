"""A turn must not freeze the asyncio event loop.

A host that serves every user from one event loop (a single uvicorn worker)
stops answering *all* of them for as long as synchronous code runs on the loop
thread.  Sync tool methods are offloaded to worker threads (see
test_sync_tool_offload.py); these tests cover the rest of what a turn keeps
off the loop: other host code, and library work that can be slow.

Each test drives a real path of the turn with a stand-in that logs the thread
it runs on and blocks with ``time.sleep``.  It checks that the work did not
run on the loop's thread and, on the main paths, that a heartbeat sharing the
loop (``loop_probe`` in conftest.py) never waited anywhere near as long as the
stand-in blocked.  Nothing here depends on how fast the machine is.
"""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from agentlys import Agentlys, APIProvider, Message
from agentlys.compaction import TokenThresholdCompaction

# How long the stand-ins block, and the longest the loop may go unanswered:
# half of it.  On the loop, a stand-in stalls the heartbeat for all of it.
BLOCK_SECONDS = 0.4
STALL_LIMIT_MS = BLOCK_SECONDS * 1000 / 2


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    # Agentlys() builds an SDK client eagerly; these tests never hit the wire.
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")


def _anthropic_reply(*content):
    """An Anthropic API response; str items become text blocks."""
    from anthropic.types import Message as AnthropicMessage

    blocks = [{"type": "text", "text": c} if isinstance(c, str) else c for c in content]
    return AnthropicMessage.model_validate(
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": blocks,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
    )


def _assert_ran_off_the_loop(thread_ids, what):
    """To be called from the test coroutine, i.e. on the event-loop thread."""
    assert thread_ids, f"{what} never ran"
    assert threading.get_ident() not in thread_ids, (
        f"{what} ran on the event-loop thread"
    )


def _serve(agent, *replies):
    """Answer the agent's Anthropic requests with canned replies, in order.

    Mocked at the SDK boundary so the provider's own request building runs.
    Returns the list the request kwargs are recorded in.
    """
    replies = list(replies)
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return replies.pop(0)

    agent.provider.client.messages.create = create
    return requests


# ---------------------------------------------------------------------------
# Tool states: every tool's __llm__() is host code
# ---------------------------------------------------------------------------


class _SlowStateTool:
    """A tool whose state costs a database round trip to compute."""

    def __init__(self):
        self.captured_on = []

    def __llm__(self):
        self.captured_on.append(threading.get_ident())
        time.sleep(BLOCK_SECONDS)
        return "orders: 42 rows"

    def count_rows(self) -> int:
        """Count the rows."""
        return 42


@pytest.mark.asyncio
async def test_tool_states_are_captured_off_the_loop_in_ask_async(loop_probe):
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    tool = _SlowStateTool()
    agent.add_tool(tool, "orders")
    requests = _serve(agent, _anthropic_reply("hello"))

    async with loop_probe as probe:
        await agent.ask_async("hi")

    assert "orders: 42 rows" in requests[0]["system"][-1]["text"]
    assert len(tool.captured_on) == 1, "the states were not computed exactly once"
    _assert_ran_off_the_loop(tool.captured_on, "__llm__()")
    probe.assert_responsive("the tool-state capture of ask_async", STALL_LIMIT_MS)


@pytest.mark.asyncio
async def test_tool_states_are_captured_off_the_loop_in_ask_stream_async(loop_probe):
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    tool = _SlowStateTool()
    agent.add_tool(tool, "orders")

    async def stream(**kwargs):
        yield {"type": "message", "message": Message(role="assistant", content="ok")}

    with patch.object(agent.provider, "fetch_stream_async", side_effect=stream):
        async with loop_probe as probe:
            async for _ in agent.ask_stream_async("hi"):
                pass

    assert "orders: 42 rows" in agent.initial_tools_states
    assert len(tool.captured_on) == 1, "the states were not computed exactly once"
    _assert_ran_off_the_loop(tool.captured_on, "__llm__()")
    probe.assert_responsive(
        "the tool-state capture of ask_stream_async", STALL_LIMIT_MS
    )


@pytest.mark.asyncio
async def test_tool_states_are_captured_off_the_loop_in_a_sub_agent(loop_probe):
    # A sub-agent starts every invocation on a fresh user message, so it
    # captures the states of all its tools each time the parent calls it.
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="explorer", provider=APIProvider.ANTHROPIC)
    tool = _SlowStateTool()
    child.add_tool(tool, "orders")
    parent.add_sub_agent(child)

    async def child_stream(**kwargs):
        yield {"type": "message", "message": Message(role="assistant", content="found")}

    with patch.object(child.provider, "fetch_stream_async", side_effect=child_stream):
        async with loop_probe as probe:
            result = await parent._call_with_signature(
                parent.functions["sub_agent__explorer"], None, prompt="look around"
            )

    assert result == "found"
    assert len(tool.captured_on) == 1, "the states were not computed exactly once"
    _assert_ran_off_the_loop(tool.captured_on, "__llm__()")
    probe.assert_responsive(
        "the tool-state capture of a sub-agent invocation", STALL_LIMIT_MS
    )


@pytest.mark.asyncio
async def test_a_capture_waiting_for_another_does_not_wait_on_the_loop(loop_probe):
    # Captures take turns — here, two invocations of one sub-agent, which
    # share its tools.  The one that waits must wait on its worker thread.
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="explorer", provider=APIProvider.ANTHROPIC)
    tool = _SlowStateTool()
    child.add_tool(tool, "orders")
    parent.add_sub_agent(child)
    explore = parent.functions["sub_agent__explorer"]

    async def child_stream(**kwargs):
        yield {"type": "message", "message": Message(role="assistant", content="found")}

    with patch.object(child.provider, "fetch_stream_async", side_effect=child_stream):
        async with loop_probe as probe:
            results = await asyncio.gather(
                parent._call_with_signature(explore, None, prompt="look here"),
                parent._call_with_signature(explore, None, prompt="look there"),
            )

    assert results == ["found", "found"]
    assert len(tool.captured_on) == 2
    _assert_ran_off_the_loop(tool.captured_on, "__llm__()")
    probe.assert_responsive("two captures of the same tools", STALL_LIMIT_MS)


@pytest.mark.asyncio
async def test_compaction_reads_tool_states_captured_off_the_loop(loop_probe):
    # Compaction runs before the turn's capture, and its summary request
    # carries the system prompt.  An agent without a snapshot (history just
    # loaded from storage) must not have the provider compute the states
    # inline for that request — nor compute them again for the turn.
    agent = Agentlys(
        provider=APIProvider.ANTHROPIC,
        compaction=TokenThresholdCompaction(token_threshold=10),
    )
    tool = _SlowStateTool()
    agent.add_tool(tool, "orders")
    agent.load_messages(
        [
            Message(role="user", content="earlier question"),
            Message(
                role="assistant",
                content="earlier answer",
                usage={"input_tokens": 1000, "output_tokens": 10},
            ),
        ]
    )
    requests = _serve(
        agent, _anthropic_reply("<summary>so far</summary>"), _anthropic_reply("ok")
    )

    async with loop_probe as probe:
        await agent.ask_async("next question")

    assert agent.messages[0].has_compaction
    summary_request, turn_request = requests
    assert "orders: 42 rows" in summary_request["system"][-1]["text"]
    assert turn_request["system"] == summary_request["system"]
    assert len(tool.captured_on) == 1, "the states were not computed exactly once"
    _assert_ran_off_the_loop(tool.captured_on, "__llm__()")
    probe.assert_responsive(
        "a compaction on an agent without a tool-state snapshot", STALL_LIMIT_MS
    )


@pytest.mark.asyncio
async def test_compaction_keeps_the_previous_snapshot_then_the_turn_takes_its_own():
    # An agent that already has a snapshot summarizes with it — that is the
    # system prompt the prompt cache holds — and only then captures the new
    # turn's states.
    class _CountingStateTool:
        def __init__(self):
            self.captures = 0

        def __llm__(self):
            self.captures += 1
            return f"capture {self.captures}"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    agent = Agentlys(
        provider=APIProvider.ANTHROPIC,
        compaction=TokenThresholdCompaction(token_threshold=150),
    )
    tool = _CountingStateTool()
    agent.add_tool(tool, "counter")
    requests = _serve(
        agent,
        _anthropic_reply("first answer"),
        _anthropic_reply("<summary>so far</summary>"),
        _anthropic_reply("second answer"),
    )

    await agent.ask_async("first question")
    agent.messages[-1].usage = {"input_tokens": 1000, "output_tokens": 10}
    await agent.ask_async("second question")

    first_turn, summary_request, second_turn = requests
    assert "capture 1" in first_turn["system"][-1]["text"]
    assert "capture 1" in summary_request["system"][-1]["text"]
    assert "capture 2" in second_turn["system"][-1]["text"]
    assert tool.captures == 2
