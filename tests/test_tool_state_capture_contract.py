"""What the tool-state capture guarantees to the host code it calls.

Tool states come from host code (``__llm__``) that reads host state: database
sessions, caches, ORM objects, often shared by the tools of several agents.
The capture runs on a worker thread so that it cannot freeze the event loop
(test_event_loop_stall.py).  A thread gives none of the following for free,
and a host whose tools share one session depends on all of it: one thread per
capture, one tool at a time, never two captures started from the same event
loop at once (states computed on the loop thread by the synchronous API
included), nothing left running once a cancelled turn has unwound.  Nor may
it take away what the loop allowed: captures started from different event
loops do not wait for each other.

The capture gave the same guarantees when it ran inline on the loop, by
construction: all these tests but the last pass on both sides of that change.
The last one is about what a thread adds — a queue in front of the capture.
"""

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from agentlys import Agentlys, APIProvider, Message


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    # Agentlys() builds an SDK client eagerly; these tests never hit the wire.
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")


class _CaptureLog:
    """Shared by the tools of a test, like a host's one session per agent."""

    def __init__(self):
        self.order = []
        self.threads = set()
        self.max_concurrent = 0
        self._concurrent = 0
        self._lock = threading.Lock()

    def enter(self, name):
        with self._lock:
            self._concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self._concurrent)
            self.order.append(name)
            self.threads.add(threading.get_ident())

    def leave(self):
        with self._lock:
            self._concurrent -= 1


class _RecordingStateTool:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __llm__(self):
        self.log.enter(self.name)
        time.sleep(0.02)  # long enough for an overlapping capture to show
        self.log.leave()
        return f"{self.name} state"

    def noop(self) -> str:
        """Do nothing."""
        return "ok"


async def _reply(**kwargs):
    return Message(role="assistant", content="ok")


@pytest.mark.asyncio
async def test_states_of_one_capture_run_in_order_on_a_single_thread():
    # Host tools share thread-affine state (one SQLAlchemy session for the
    # whole agent): their __llm__() must not be fanned out over threads.
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    log = _CaptureLog()
    for name in ("first", "second", "third"):
        agent.add_tool(_RecordingStateTool(name, log), name)

    with patch.object(agent.provider, "fetch_async", side_effect=_reply):
        await agent.ask_async("hi")

    assert log.order == ["first", "second", "third"]
    assert len(log.threads) == 1
    assert log.max_concurrent == 1
    states = agent.initial_tools_states
    assert states.index("first state") < states.index("second state")
    assert states.index("second state") < states.index("third state")


@pytest.mark.asyncio
async def test_tool_states_read_the_context_variables_of_the_turn():
    # Hosts keep what a request is about in context variables (who is asking,
    # on which session): __llm__() must find the turn's, wherever it runs.
    asking = contextvars.ContextVar("asking", default=None)
    seen = []

    class _ReadsTheRequestContext:
        def __llm__(self):
            seen.append(asking.get())
            return "state"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_tool(_ReadsTheRequestContext(), "tool")
    agent.provider.fetch_async = _reply

    asking.set("alice")
    await agent.ask_async("hi")

    assert seen == ["alice"]


@pytest.mark.asyncio
async def test_parallel_sub_agent_invocations_never_capture_shared_tools_at_once():
    # Every invocation of a sub-agent runs on a shallow copy that shares the
    # template's tool instances.  The parent may call it several times in one
    # turn; their captures must still take turns.
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="explorer", provider=APIProvider.ANTHROPIC)
    log = _CaptureLog()
    child.add_tool(_RecordingStateTool("shared", log), "shared")
    parent.add_sub_agent(child)
    explore = parent.functions["sub_agent__explorer"]

    async def child_stream(**kwargs):
        yield {"type": "message", "message": Message(role="assistant", content="found")}

    with patch.object(child.provider, "fetch_stream_async", side_effect=child_stream):
        results = await asyncio.gather(
            *[
                parent._call_with_signature(explore, None, prompt=f"task {i}")
                for i in range(4)
            ]
        )

    assert results == ["found"] * 4
    assert len(log.order) == 4
    assert log.max_concurrent == 1


@pytest.mark.asyncio
async def test_captures_of_unrelated_agents_on_one_event_loop_never_overlap():
    # Agents that know nothing of each other can still be built on the same
    # host state: two sub-agents on one database session, say.  The library
    # cannot tell, so captures started from the same event loop take turns —
    # as they did when each of them held that loop for as long as it lasted.
    log = _CaptureLog()
    agents = []
    for name in ("first", "second", "third"):
        agent = Agentlys(provider=APIProvider.ANTHROPIC)
        agent.add_tool(_RecordingStateTool(name, log), name)
        agent.provider.fetch_async = _reply
        agents.append(agent)

    await asyncio.gather(*(agent.ask_async("hi") for agent in agents))

    assert sorted(log.order) == ["first", "second", "third"]
    assert log.max_concurrent == 1


def test_captures_started_from_different_event_loops_do_not_wait_for_each_other():
    # They never did: a loop only ever held up its own captures.  Each
    # __llm__() here returns once the other capture is inside its own.
    both_inside = threading.Barrier(2)
    met = []

    class _MeetsTheOtherCapture:
        def __llm__(self):
            try:
                both_inside.wait(5)
            except threading.BrokenBarrierError:
                met.append(False)
            else:
                met.append(True)
            return "state"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    def turn_on_its_own_loop():
        agent = Agentlys(provider=APIProvider.ANTHROPIC)
        agent.add_tool(_MeetsTheOtherCapture(), "tool")
        agent.provider.fetch_async = _reply
        asyncio.run(agent.ask_async("hi"))

    with ThreadPoolExecutor(max_workers=2) as threads:
        for turn in [threads.submit(turn_on_its_own_loop) for _ in range(2)]:
            turn.result(30)

    assert met == [True, True]


@pytest.mark.asyncio
async def test_a_tool_state_can_wait_for_a_turn_run_on_a_helper_thread():
    # A __llm__() that asks another agent for its text, on a helper thread
    # with a loop of its own.  That turn's capture must not wait for the one
    # that is waiting for it: it did not when captures ran on their loop.
    log = _CaptureLog()
    joined = []

    async def summary(**kwargs):
        return Message(role="assistant", content="a summary")

    class _SummarizedByAnotherAgent:
        def __llm__(self):
            answers = []

            def ask_the_other_agent():
                other = Agentlys(provider=APIProvider.ANTHROPIC)
                other.add_tool(_RecordingStateTool("other", log), "other")
                other.provider.fetch_async = summary
                answers.append(asyncio.run(other.ask_async("summarize")).content)

            helper = threading.Thread(target=ask_the_other_agent)
            helper.start()
            helper.join(5)
            joined.append(not helper.is_alive())
            return answers[0] if answers else "no summary"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_tool(_SummarizedByAnotherAgent(), "summarized")
    agent.provider.fetch_async = _reply

    await agent.ask_async("hi")

    assert joined == [True], "the other agent's capture waited for this one"
    assert log.order == ["other"]
    assert "a summary" in agent.initial_tools_states


@pytest.mark.parametrize(
    "compute_states",
    [
        lambda agent: agent.refresh_tools_states(),
        lambda agent: agent.initial_tools_states,  # what a provider reads
        lambda agent: agent._capture_tools_states(),
    ],
    ids=["refresh_tools_states", "initial_tools_states", "_capture_tools_states"],
)
@pytest.mark.asyncio
async def test_states_computed_on_the_loop_thread_wait_for_a_capture_in_progress(
    compute_states,
):
    # The synchronous API computes on the calling thread.  Called on the loop
    # thread it could not overlap a capture when captures ran there too: two
    # agents sharing a tool must still never be inside its __llm__() together.
    class _SlowSharedStateTool:
        def __init__(self):
            self.log = _CaptureLog()
            self.entered = threading.Event()

        def __llm__(self):
            self.log.enter("shared")
            self.entered.set()
            time.sleep(0.2)
            self.log.leave()
            return "shared state"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    tool = _SlowSharedStateTool()
    capturing = Agentlys(provider=APIProvider.ANTHROPIC)
    capturing.add_tool(tool, "shared")
    capturing.provider.fetch_async = _reply
    other = Agentlys(provider=APIProvider.ANTHROPIC)
    other.add_tool(tool, "shared")

    turn = asyncio.create_task(capturing.ask_async("hi"))
    for _ in range(500):  # until the capture is inside __llm__(), 5 s at most
        if tool.entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert tool.entered.is_set(), "the capture never started"

    compute_states(other)
    await turn

    assert len(tool.log.order) == 2
    assert tool.log.max_concurrent == 1


def test_a_tool_state_can_read_the_states_of_another_agent():
    # Through the synchronous API, from inside __llm__(): on the worker thread
    # of a capture, and on the loop thread when the states that call it are
    # themselves computed there.  Neither may wait for the turn it is part of.
    class _QuotesAnotherAgent:
        def __init__(self, other):
            self.other = other

        def __llm__(self):
            return f"the other agent sees: {self.other.initial_tools_states}"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    inner = Agentlys(provider=APIProvider.ANTHROPIC)
    inner.add_tool(_RecordingStateTool("inner", _CaptureLog()), "inner")
    outer = Agentlys(provider=APIProvider.ANTHROPIC)
    outer.add_tool(_QuotesAnotherAgent(inner), "outer")
    outer.provider.fetch_async = _reply
    seen = []

    async def on_the_loop_thread_then_on_a_worker():
        outer.refresh_tools_states()
        seen.append(outer.initial_tools_states)
        await outer.ask_async("hi")
        seen.append(outer.initial_tools_states)

    # On a daemon thread: a read that waits for itself fails the test instead
    # of hanging it.
    scenario = threading.Thread(
        target=lambda: asyncio.run(on_the_loop_thread_then_on_a_worker()),
        daemon=True,
    )
    scenario.start()
    scenario.join(5)

    assert not scenario.is_alive(), "a nested read waited for the turn it is part of"
    assert len(seen) == 2
    assert all("inner state" in states for states in seen)


@pytest.mark.asyncio
async def test_failing_tool_state_fails_the_turn_before_any_request():
    class _BrokenStateTool:
        def __llm__(self):
            raise RuntimeError("database is down")

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_tool(_BrokenStateTool(), "broken")

    with patch.object(agent.provider, "fetch_async", side_effect=_reply) as fetch:
        with pytest.raises(RuntimeError, match="database is down"):
            await agent.ask_async("hi")

    assert fetch.await_count == 0


@pytest.mark.asyncio
async def test_cancelled_turn_leaves_no_tool_state_running():
    # A thread cannot be interrupted.  If the capture is handed to one and the
    # turn is cancelled meanwhile, __llm__() must not keep reading host state
    # after the host saw the turn end and started closing that state.
    class _TrackedStateTool:
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

    async def never_answers(**kwargs):
        await asyncio.sleep(3600)

    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    tool = _TrackedStateTool()
    agent.add_tool(tool, "tracked")

    with patch.object(agent.provider, "fetch_async", side_effect=never_answers):
        turn = asyncio.create_task(agent.ask_async("hi"))
        for _ in range(500):  # until the capture has started, 5 s at most
            if tool.started.is_set():
                break
            await asyncio.sleep(0.01)
        assert tool.started.is_set(), "the capture never started"
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    assert tool.running is False


def test_closing_loop_leaves_no_tool_state_running():
    # A loop that closes cancels every task it still has, whatever they are
    # waiting for.  That is one more cancellation: the turn must not unwind,
    # and the host close what __llm__() is reading, before it has returned.
    events = []
    started = threading.Event()

    class _ReadsWhatTheHostCloses:
        def __llm__(self):
            started.set()
            time.sleep(0.3)
            events.append("state returned")
            return "state"

        def noop(self) -> str:
            """Do nothing."""
            return "ok"

    async def turn(agent):
        try:
            await agent.ask_async("hi")
        finally:
            events.append("host closed its resources")

    async def main():
        agent = Agentlys(provider=APIProvider.ANTHROPIC)
        agent.add_tool(_ReadsWhatTheHostCloses(), "tool")
        agent.provider.fetch_async = _reply
        running = asyncio.create_task(turn(agent))
        for _ in range(500):  # until the capture has started, 5 s at most
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "the capture never started"
        return running  # asyncio.run() now closes the loop under the turn

    asyncio.run(main())

    assert events == ["state returned", "host closed its resources"]


@pytest.mark.asyncio
async def test_cancelled_turn_does_not_wait_for_a_capture_that_never_started():
    # Waiting is for host code that is running.  A capture still queued behind
    # a busy executor is dropped instead: the cancellation goes through at
    # once, and __llm__() is not called later behind the host's back.
    log = _CaptureLog()
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_tool(_RecordingStateTool("queued", log), "queued")

    executor = ThreadPoolExecutor(max_workers=1)
    asyncio.get_running_loop().set_default_executor(executor)
    busy = threading.Event()
    release = threading.Event()

    def occupy_the_only_worker():
        busy.set()
        release.wait(10)

    occupied = executor.submit(occupy_the_only_worker)
    try:
        assert busy.wait(5)
        with patch.object(agent.provider, "fetch_async", side_effect=_reply):
            turn = asyncio.create_task(agent.ask_async("hi"))
            await asyncio.sleep(0.05)  # the capture is now queued
            turn.cancel()
            finished, _ = await asyncio.wait([turn], timeout=5)
    finally:
        release.set()
    occupied.result(5)
    executor.shutdown(wait=True)  # the queued capture has had its turn
    await asyncio.sleep(0.05)

    assert finished, "the cancellation waited for a capture that had not started"
    assert turn.cancelled()
    assert log.order == []
