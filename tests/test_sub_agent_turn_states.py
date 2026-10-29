"""Sub-agent tool states captured at the start of the parent's turn.

``add_sub_agent(..., capture_states_at_turn_start=True)``: the parent's async
turn captures the sub-agent's states before any request, and every invocation
of that turn sends them, instead of capturing at each invocation in the middle
of a tool batch (where its tools may share host state with the batch's other
tools).  What a turn captured lives in the turn's context, never on the
template: one template can serve several parents, users and turns at once.
"""

import asyncio
import contextvars
import threading
import time
from unittest.mock import patch

import pytest

from agentlys import Agentlys
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import APIProvider


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    # Agentlys() builds an SDK client eagerly; these tests never hit the wire.
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")


def _block(*states):
    tools = "\n".join(states)
    return f"## Initial Tools States\n{tools}\n--- End of Initial Tools States ---"


class _StateTool:
    """Counts its ``__llm__`` calls and records them in a shared trace."""

    def __init__(self, trace, name):
        self.trace = trace
        self.name = name
        self.calls = 0

    def __llm__(self):
        self.calls += 1
        self.trace.append(f"{self.name} states")
        return f"{self.name} #{self.calls}"

    def noop(self) -> str:
        """Do nothing."""
        return "ok"


def _agent(name=None):
    return Agentlys(name=name, provider=APIProvider.ANTHROPIC)


def _calls(name, count, tag):
    """An assistant message calling sub-agent ``name`` ``count`` times at once."""
    return Message(
        role="assistant",
        parts=[
            MessagePart(
                type="function_call",
                function_call={
                    "name": f"sub_agent__{name}",
                    "arguments": {"prompt": "go"},
                },
                function_call_id=f"call_{tag}_{i}",
            )
            for i in range(count)
        ],
    )


def _answer(text="done"):
    return Message(role="assistant", content=text)


class _Script:
    """Stands in for every model: ``replies[agent name]`` says what it answers.

    A reply function gets the agent (the invocation's copy for a sub-agent)
    and returns its next message.  Every request and the states block it
    carried are recorded, per agent name.
    """

    def __init__(self, trace, replies):
        self.trace = trace
        self.replies = replies
        self.sent = {}

    async def stream(self, provider, **kwargs):
        chat = provider.chat
        name = chat.name or "parent"
        self.trace.append(f"{name} request")
        self.sent.setdefault(name, []).append(chat.initial_tools_states)
        yield {"type": "message", "message": self.replies[name](chat)}


def _delegate(name, count=1):
    """Parent reply: call ``name`` ``count`` times, then answer."""
    turns = []

    def reply(chat):
        if chat.messages[-1].role == "function":
            return _answer()
        turns.append(None)
        return _calls(name, count, len(turns))

    return reply


async def _run(parent, script, *questions):
    with patch.object(
        type(parent.provider),
        "fetch_stream_async",
        autospec=True,
        side_effect=script.stream,
    ):
        for question in questions:
            async for _ in parent.run_conversation_stream_async(question):
                pass


def test_sub_agents_lists_the_templates_by_function_name():
    parent, child = _agent(), _agent("explorer")
    parent.add_sub_agent(child)

    assert parent.sub_agents == {"sub_agent__explorer": child}
    parent.sub_agents.clear()  # a copy: the registry is untouched
    assert "sub_agent__explorer" in parent.sub_agents


@pytest.mark.asyncio
async def test_captured_before_any_request_and_sent_by_every_invocation_of_the_turn():
    parent, child = _agent(), _agent("explorer")
    trace = []
    child.add_tool(_StateTool(trace, "child"), "probe")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)
    script = _Script(
        trace,
        {"parent": _delegate("explorer", 2), "explorer": lambda c: _answer("found")},
    )

    await _run(parent, script, "first", "second")

    # Once per parent turn, before its first request: never inside the batch.
    assert trace[0] == "child states"
    assert trace.count("child states") == 2
    second_turn = trace.index("child states", 1)
    assert trace[:second_turn].count("explorer request") == 2
    child_block = "### _StateTool-probe\nchild #{}"
    assert (
        script.sent["explorer"]
        == [_block(child_block.format(1))] * 2 + [_block(child_block.format(2))] * 2
    )
    # Nothing of the turn is left on the template.
    assert child._initial_tools_states is None


@pytest.mark.asyncio
async def test_without_the_option_each_invocation_captures():
    parent, child = _agent(), _agent("explorer")
    trace = []
    child.add_tool(_StateTool(trace, "child"), "probe")
    parent.add_sub_agent(child)
    script = _Script(
        trace, {"parent": _delegate("explorer", 2), "explorer": lambda c: _answer()}
    )

    await _run(parent, script, "first")

    assert trace[0] == "parent request"
    assert trace.count("child states") == 2


@pytest.mark.asyncio
async def test_a_pin_wins_even_when_set_after_the_turn_started():
    parent, child = _agent(), _agent("explorer")
    trace = []
    child.add_tool(_StateTool(trace, "child"), "probe")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)

    def parent_reply(chat):
        # The host pins between the turn's capture and the invocation.
        parent.sub_agents["sub_agent__explorer"].pin_tools_states("pinned")
        return _delegate("explorer")(chat)

    script = _Script(trace, {"parent": parent_reply, "explorer": lambda c: _answer()})
    await _run(parent, script, "first")
    assert script.sent["explorer"] == ["pinned"]

    trace.clear()
    await _run(parent, script, "second")  # pinned before the turn: not captured
    assert "child states" not in trace
    assert script.sent["explorer"] == ["pinned", "pinned"]


@pytest.mark.asyncio
async def test_a_failing_state_is_left_to_the_invocation_and_does_not_fail_the_turn(
    caplog,
):
    class _FailsOnce:
        def __init__(self):
            self.calls = 0

        def __llm__(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("unreachable")
            return "recovered"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    parent, child = _agent(), _agent("explorer")
    tool = _FailsOnce()
    child.add_tool(tool, "flaky")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)
    script = _Script(
        [], {"parent": _delegate("explorer"), "explorer": lambda c: _answer()}
    )

    await _run(parent, script, "hi")

    assert "Tool states of sub-agent 'explorer' failed" in caplog.text
    # Captured again by the invocation, as without the option.
    assert tool.calls == 2
    assert script.sent["explorer"] == [_block("### _FailsOnce-flaky\nrecovered")]


@pytest.mark.asyncio
async def test_a_turn_whose_capture_fails_does_not_send_the_previous_turns_states():
    class _GoodThenBroken:
        def __init__(self):
            self.calls = 0

        def __llm__(self):
            self.calls += 1
            if self.calls == 1:
                return "turn one"
            raise RuntimeError("down")

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    parent, child = _agent(), _agent("explorer")
    child.add_tool(_GoodThenBroken(), "t")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)
    script = _Script(
        [], {"parent": _delegate("explorer"), "explorer": lambda c: _answer()}
    )

    await _run(parent, script, "first", "second")

    # Turn two's invocation captured for itself (and failed there, as it would
    # have without the option): it never sent turn one's block.
    assert script.sent["explorer"] == [_block("### _GoodThenBroken-t\nturn one")]


@pytest.mark.asyncio
async def test_concurrent_turns_sharing_a_template_never_read_each_others_states():
    user = contextvars.ContextVar("user")

    class _UserMemory:
        def __llm__(self):
            return f"memory of {user.get()}"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    child = _agent("explorer")  # one template, registered on two parents
    child.add_tool(_UserMemory(), "memory")
    sent = {}
    both_captured = asyncio.Event()
    started = []

    async def stream(provider, **kwargs):
        chat = provider.chat
        if chat.name == "explorer":
            sent.setdefault(user.get(), []).append(chat.initial_tools_states)
            yield {"type": "message", "message": _answer()}
            return
        if chat.messages[-1].role == "function":
            yield {"type": "message", "message": _answer()}
            return
        # Both turns have captured before either invokes the sub-agent.
        started.append(user.get())
        if len(started) == 2:
            both_captured.set()
        await both_captured.wait()
        yield {"type": "message", "message": _calls("explorer", 1, user.get())}

    async def turn(name):
        parent = _agent()
        parent.add_sub_agent(child, capture_states_at_turn_start=True)
        user.set(name)
        async for _ in parent.run_conversation_stream_async("hi"):
            pass

    with patch.object(
        type(child.provider), "fetch_stream_async", autospec=True, side_effect=stream
    ):
        await asyncio.gather(turn("alice"), turn("bob"))

    assert sent == {
        "alice": [_block("### _UserMemory-memory\nmemory of alice")],
        "bob": [_block("### _UserMemory-memory\nmemory of bob")],
    }


@pytest.mark.asyncio
async def test_nested_sub_agents_are_captured_at_the_root_turn_start_only():
    parent, child, grandchild = _agent(), _agent("child"), _agent("grand")
    trace = []
    grandchild.add_tool(_StateTool(trace, "grand"), "g")
    child.pin_tools_states("child pinned by the host")
    child.add_sub_agent(grandchild, capture_states_at_turn_start=True)
    parent.add_sub_agent(child, capture_states_at_turn_start=True)
    script = _Script(
        trace,
        {
            "parent": _delegate("child", 2),
            "child": _delegate("grand"),
            "grand": lambda c: _answer(),
        },
    )

    await _run(parent, script, "hi")

    # Once, before the parent's first request; never inside the child's
    # invocations, pinned as the child is.
    assert trace[0] == "grand states"
    assert trace.count("grand states") == 1
    assert script.sent["grand"] == [_block("### _StateTool-g\ngrand #1")] * 2


@pytest.mark.asyncio
async def test_a_turn_that_no_longer_captures_a_sub_agent_does_not_send_older_states():
    parent, child = _agent(), _agent("explorer")
    trace = []
    child.add_tool(_StateTool(trace, "child"), "probe")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)
    script = _Script(
        trace, {"parent": _delegate("explorer"), "explorer": lambda c: _answer()}
    )
    await _run(parent, script, "first")

    parent.remove_sub_agent("explorer")
    parent.add_sub_agent(child)  # without the option now
    await _run(parent, script, "second")  # same task, same context

    assert script.sent["explorer"] == [
        _block("### _StateTool-probe\nchild #1"),
        _block("### _StateTool-probe\nchild #2"),
    ]


@pytest.mark.asyncio
async def test_an_agent_registered_as_its_own_sub_agent_still_runs_its_turn():
    agent = _agent("helper")
    trace = []
    agent.add_tool(_StateTool(trace, "helper"), "t")
    agent.add_sub_agent(agent, name="helper", capture_states_at_turn_start=True)
    script = _Script(trace, {"helper": lambda c: _answer()})

    await _run(agent, script, "hi")  # no RecursionError walking the cycle

    assert script.sent["helper"] == [_block("### _StateTool-t\nhelper #1")]


@pytest.mark.asyncio
async def test_turn_start_captures_of_concurrent_turns_never_overlap():
    # Two parents on one loop, sub-agents built on the same host state: their
    # turn-start captures take turns, as every capture of a loop does.
    lock = threading.Lock()
    concurrent = [0, 0]  # now, max
    both_in = threading.Event()

    class _Shared:
        def __llm__(self):
            with lock:
                concurrent[0] += 1
                concurrent[1] = max(concurrent)
                if concurrent[0] == 2:
                    both_in.set()
            # Unserialized, the other turn's capture comes in meanwhile.
            both_in.wait(0.5)
            with lock:
                concurrent[0] -= 1
            return "state"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    async def answer(provider, **kwargs):
        yield {"type": "message", "message": _answer()}

    async def turn():
        parent, child = _agent(), _agent("explorer")
        child.add_tool(_Shared(), "shared")
        parent.add_sub_agent(child, capture_states_at_turn_start=True)
        async for _ in parent.run_conversation_stream_async("hi"):
            pass

    with patch.object(
        type(_agent().provider), "fetch_stream_async", autospec=True, side_effect=answer
    ):
        await asyncio.gather(turn(), turn())

    assert concurrent[1] == 1


@pytest.mark.asyncio
async def test_a_cancelled_turn_leaves_no_sub_agent_state_running():
    class _Tracked:
        def __init__(self):
            self.started = threading.Event()
            self.running = False

        def __llm__(self):
            self.running = True
            self.started.set()
            time.sleep(0.3)
            self.running = False
            return "state"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    parent, child = _agent(), _agent("explorer")
    tool = _Tracked()
    child.add_tool(tool, "tracked")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)

    async def consume():
        async for _ in parent.run_conversation_stream_async("hi"):
            pass

    turn = asyncio.create_task(consume())
    for _ in range(500):  # until the capture has started, 5 s at most
        if tool.started.is_set():
            break
        await asyncio.sleep(0.01)
    assert tool.started.is_set(), "the capture never started"
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert tool.running is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [True, False])
async def test_a_run_resumed_from_loaded_history_captures_at_its_start(streamed):
    # Hosts load the stored history, user message included, and run without
    # a new message: that run is a turn too.
    parent, child = _agent(), _agent("explorer")
    trace = []
    child.add_tool(_StateTool(trace, "child"), "probe")
    parent.add_sub_agent(child, capture_states_at_turn_start=True)
    parent.load_messages([Message(role="user", content="what is in there?")])
    script = _Script(
        trace, {"parent": _delegate("explorer", 2), "explorer": lambda c: _answer()}
    )

    async def fetch(provider, **kwargs):
        async for chunk in script.stream(provider, **kwargs):
            return chunk["message"]

    with (
        patch.object(
            type(parent.provider),
            "fetch_stream_async",
            autospec=True,
            side_effect=script.stream,
        ),
        patch.object(
            type(parent.provider), "fetch_async", autospec=True, side_effect=fetch
        ),
    ):
        run = (
            parent.run_conversation_stream_async()
            if streamed
            else parent.run_conversation_async()
        )
        async for _ in run:
            pass

    assert trace[0] == "child states"
    assert trace.count("child states") == 1
    assert script.sent["explorer"] == [_block("### _StateTool-probe\nchild #1")] * 2
