"""Per-call isolation seam: ``__agentlys_call_scope__``.

A turn's tool calls run concurrently — sync tools on worker threads, async
tools as interleaved tasks — yet share one tool instance. Instance state that
isn't safe to share (a DB session, an open transaction, a non-thread-safe
client) then races across calls.

When a tool exposes ``__agentlys_call_scope__`` (a context manager), each call
runs on the instance it yields, so concurrent calls get isolated state. These
tests drive the seam through ``_call_with_signature`` — the chokepoint every
dispatch path funnels through.
"""

import asyncio
import contextlib
import copy
import threading

import pytest

from agentlys import Agentlys
from agentlys.chat import StopLoopException


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    # Agentlys() builds an OpenAI client eagerly; these tests never hit the wire.
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")


class _ScopedTool:
    """Hands each call its own ``session`` via the scope; records lifecycle."""

    def __init__(self):
        self.session = None  # shared state that must NOT be reused across calls
        self.entered = []
        self.exited = []
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def __agentlys_call_scope__(self):
        scoped = copy.copy(self)
        scoped.session = object()  # a distinct session per call
        with self._lock:
            self.entered.append(scoped.session)
        try:
            yield scoped
        finally:
            with self._lock:
                self.exited.append(scoped.session)

    def work(self, **kwargs):
        # Runs on the scoped copy, so it sees that copy's own session.
        assert self.session is not None, "tool ran without a scoped session"
        return self.session

    async def awork(self, **kwargs):
        await asyncio.sleep(0.01)
        assert self.session is not None
        return self.session

    def boom(self, **kwargs):
        raise ValueError("kaboom")

    def stop(self, **kwargs):
        raise StopLoopException("done")


class _ThreadRecordingTool:
    """Records the thread on which scope enter/exit and the body each run.

    ``threads`` is a shared dict: the scope yields a shallow copy, so the copy
    (which runs the body) and the original (which runs enter/exit) write into
    the same dict.
    """

    def __init__(self):
        self.session = None
        self.threads = {}

    @contextlib.contextmanager
    def __agentlys_call_scope__(self):
        scoped = copy.copy(self)
        scoped.session = object()
        self.threads["enter"] = threading.get_ident()
        try:
            yield scoped
        finally:
            self.threads["exit"] = threading.get_ident()

    def work(self, **kwargs):
        self.threads["body"] = threading.get_ident()
        return self.session


@pytest.mark.asyncio
async def test_sync_scope_lifecycle_and_body_share_one_worker_thread():
    # A thread-affine resource (eg a SQLAlchemy Session) must be opened, used
    # and closed on the same thread. For sync tools that means the scope's
    # enter/exit run on the worker thread alongside the offloaded body — never
    # on the event-loop thread.
    chat = Agentlys()
    tool = _ThreadRecordingTool()
    loop_thread = threading.get_ident()

    await chat._call_with_signature(tool.work, None)

    t = tool.threads
    assert t["enter"] == t["body"] == t["exit"]
    assert t["body"] != loop_thread, "sync body must run off the event loop"


class _PlainTool:
    """No scope → calls run on the original instance (current behaviour)."""

    def __init__(self):
        self.marker = "original"

    def work(self, **kwargs):
        return self.marker


@pytest.mark.asyncio
async def test_scope_runs_method_on_scoped_instance():
    chat = Agentlys()
    tool = _ScopedTool()
    session = await chat._call_with_signature(tool.work, None)
    assert session is not None
    assert tool.session is None, "the original instance must stay untouched"
    assert len(tool.entered) == len(tool.exited) == 1


@pytest.mark.asyncio
async def test_parallel_sync_calls_get_isolated_sessions():
    chat = Agentlys()
    tool = _ScopedTool()
    n = 12
    sessions = await asyncio.gather(
        *[chat._call_with_signature(tool.work, None) for _ in range(n)]
    )
    assert len({id(s) for s in sessions}) == n, "calls shared a session"
    assert len(tool.entered) == len(tool.exited) == n


@pytest.mark.asyncio
async def test_parallel_async_calls_get_isolated_sessions():
    chat = Agentlys()
    tool = _ScopedTool()
    n = 12
    sessions = await asyncio.gather(
        *[chat._call_with_signature(tool.awork, None) for _ in range(n)]
    )
    assert len({id(s) for s in sessions}) == n


@pytest.mark.asyncio
async def test_tool_without_scope_is_unaffected():
    chat = Agentlys()
    tool = _PlainTool()
    assert await chat._call_with_signature(tool.work, None) == "original"


@pytest.mark.asyncio
async def test_exception_propagates_and_scope_exits():
    chat = Agentlys()
    tool = _ScopedTool()
    with pytest.raises(ValueError):
        await chat._call_with_signature(tool.boom, None)
    assert len(tool.exited) == 1, "scope must exit (cleanup) even on error"


@pytest.mark.asyncio
async def test_stoploop_propagates_and_scope_exits():
    chat = Agentlys()
    tool = _ScopedTool()
    with pytest.raises(StopLoopException):
        await chat._call_with_signature(tool.stop, None)
    assert len(tool.exited) == 1
