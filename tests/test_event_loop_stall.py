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
import base64
import io
import threading
import time
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from agentlys import Agentlys, APIProvider, Message, MessagePart
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


def _tool_use(name, call_id, **arguments):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


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


# ---------------------------------------------------------------------------
# Images: PIL encodes each one for the request that carries it
# ---------------------------------------------------------------------------


def _slow_to_encode(image, encoded_on):
    """Make a PIL image whose encoding blocks, logging the thread that does it."""
    encode = image.save

    def save(fp, format=None, **params):
        encoded_on.append(threading.get_ident())
        time.sleep(BLOCK_SECONDS)
        return encode(fp, format=format, **params)

    image.save = save
    return image


def _stored_image(encoded_on):
    """An image as a host restores it from storage: bytes, opened lazily."""
    buffer = io.BytesIO()
    PILImage.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    image = PILImage.open(io.BytesIO(buffer.getvalue()))
    return MessagePart(type="image", image=_slow_to_encode(image, encoded_on))


def _images_in(request_message):
    return [block for block in request_message["content"] if block["type"] == "image"]


@pytest.mark.asyncio
async def test_an_image_returned_by_a_tool_is_encoded_off_the_loop(loop_probe):
    # The tool itself is instant: the cost is encoding its image for the
    # request that sends the result back to the model.
    encoded_on = []
    chart = _slow_to_encode(PILImage.new("RGB", (32, 32), "red"), encoded_on)

    def render_chart():
        """Render the chart."""
        return chart

    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(render_chart)
    requests = _serve(
        agent,
        _anthropic_reply(_tool_use("render_chart", "toolu_01")),
        _anthropic_reply("here is your chart"),
    )

    async with loop_probe as probe:
        async for _ in agent.run_conversation_async("a chart please"):
            pass

    (tool_result,) = requests[1]["messages"][-1]["content"]
    (sent,) = _images_in(tool_result)
    decoded = PILImage.open(io.BytesIO(base64.b64decode(sent["source"]["data"])))
    assert decoded.size == (32, 32)
    assert len(encoded_on) == 1
    _assert_ran_off_the_loop(encoded_on, "the encoding of the tool's image")
    probe.assert_responsive("the request carrying a tool's image", STALL_LIMIT_MS)


@pytest.mark.asyncio
async def test_stored_images_are_encoded_off_the_loop_and_only_once(loop_probe):
    # A host that keeps history in a database rebuilds its images on every
    # run, so the first request of a turn finds none of them encoded.
    encoded_on = []
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.load_messages(
        [
            Message(
                role="user",
                parts=[
                    MessagePart(type="text", content="what do these show?"),
                    _stored_image(encoded_on),
                    _stored_image(encoded_on),
                ],
            ),
            Message(role="assistant", content="two red squares"),
        ]
    )
    requests = _serve(agent, _anthropic_reply("ok"), _anthropic_reply("you're welcome"))

    async with loop_probe as probe:
        await agent.ask_async("and the second one?")

    assert len(_images_in(requests[0]["messages"][0])) == 2
    assert len(encoded_on) == 2
    _assert_ran_off_the_loop(encoded_on, "the encoding of the stored images")
    probe.assert_responsive("the first request of a turn with images", STALL_LIMIT_MS)

    await agent.ask_async("thanks")

    assert len(_images_in(requests[1]["messages"][0])) == 2
    assert len(encoded_on) == 2, "an image already encoded was encoded again"


@pytest.mark.asyncio
async def test_an_image_held_by_two_agents_is_not_encoded_by_both_at_once():
    # One history handed to several agents: each encodes on its own worker
    # thread, and PIL does not support two threads working on one image.
    encoded_on = []
    shared = Message(role="user", parts=[_stored_image(encoded_on)])
    agents = []
    for _ in range(2):
        agent = Agentlys(provider=APIProvider.ANTHROPIC)
        agent.load_messages([shared, Message(role="assistant", content="a square")])
        _serve(agent, _anthropic_reply("ok"))
        agents.append(agent)

    await asyncio.gather(*(agent.ask_async("what colour?") for agent in agents))

    assert len(encoded_on) == 1, "both agents encoded the image they share"


@pytest.mark.asyncio
async def test_images_are_encoded_off_the_loop_before_a_streamed_request():
    encoded_on = []
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.load_messages(
        [
            Message(role="user", parts=[_stored_image(encoded_on)]),
            Message(role="assistant", content="a red square"),
        ]
    )

    async def stream(**kwargs):
        agent.provider._prepare_request_params()  # what every provider starts with
        yield {"type": "message", "message": Message(role="assistant", content="ok")}

    with patch.object(agent.provider, "fetch_stream_async", side_effect=stream):
        async for _ in agent.ask_stream_async("what colour?"):
            pass

    assert len(encoded_on) == 1
    _assert_ran_off_the_loop(encoded_on, "the encoding of the stored image")


@pytest.mark.asyncio
async def test_images_are_encoded_off_the_loop_before_a_compaction():
    # The summary request is the conversation's own request plus a prompt: it
    # carries the images as well, and comes before the turn's request.
    encoded_on = []
    agent = Agentlys(
        provider=APIProvider.ANTHROPIC,
        compaction=TokenThresholdCompaction(token_threshold=10),
    )
    agent.load_messages(
        [
            Message(role="user", parts=[_stored_image(encoded_on)]),
            Message(
                role="assistant",
                content="a red square",
                usage={"input_tokens": 1000, "output_tokens": 10},
            ),
        ]
    )
    requests = _serve(
        agent, _anthropic_reply("<summary>so far</summary>"), _anthropic_reply("ok")
    )

    await agent.ask_async("next question")

    assert len(_images_in(requests[0]["messages"][0])) == 1
    assert len(encoded_on) == 1
    _assert_ran_off_the_loop(encoded_on, "the encoding of the stored image")


@pytest.mark.asyncio
async def test_an_image_that_cannot_be_encoded_still_fails_in_the_request():
    # Encoding ahead of the request is an optimization: an image PIL cannot
    # write surfaces where it always has, from the provider call.
    def save(fp, format=None, **params):
        raise OSError("cannot write this image")

    broken = PILImage.new("RGB", (8, 8))
    broken.save = save
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.load_messages(
        [Message(role="user", parts=[MessagePart(type="image", image=broken)])]
    )
    requests = _serve(agent, _anthropic_reply("unreachable"))

    with patch.object(
        agent.provider, "fetch_async", wraps=agent.provider.fetch_async
    ) as fetch:
        with pytest.raises(ValueError, match="Failed to convert image to base64"):
            await agent.ask_async("what is this?")

    assert fetch.await_count == 1, "the error did not come from the provider call"
    assert requests == []
