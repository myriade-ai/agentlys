"""Sync tools must not freeze the asyncio event loop.

Regression: `_call_with_signature` used to invoke sync tool functions directly
on the event loop thread. A slow sync tool (eg a blocking warehouse driver
call) therefore froze every other in-flight request until it returned.
The fix routes sync calls through `asyncio.to_thread`; this test verifies the
loop keeps ticking while a slow sync tool is running.
"""

import asyncio
import time

import pytest

from agentlys import Agentlys


BLOCK_SECONDS = 0.4
HEARTBEAT_PERIOD = 0.02


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    """Agentlys() constructs an OpenAI client eagerly. These tests never hit
    the wire — just make the constructor happy."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")


def slow_sync_tool() -> str:
    """Stand-in for a hung driver call."""
    time.sleep(BLOCK_SECONDS)
    return "ok"


async def _heartbeat(stop: asyncio.Event) -> int:
    """Count loop ticks that fire while the tool is running."""
    ticks = 0
    while not stop.is_set():
        ticks += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_PERIOD)
        except asyncio.TimeoutError:
            pass
    return ticks


@pytest.mark.asyncio
async def test_sync_tool_does_not_block_event_loop():
    chat = Agentlys()

    stop = asyncio.Event()
    hb = asyncio.create_task(_heartbeat(stop))
    await asyncio.sleep(0)  # let heartbeat get its first tick in

    result = await chat._call_with_signature(slow_sync_tool, None)

    stop.set()
    ticks = await hb

    assert result == "ok"

    # A healthy loop should fire roughly BLOCK_SECONDS / HEARTBEAT_PERIOD ticks
    # while the tool is sleeping. Pre-fix this is 0 or 1; we require at least
    # half of the expected ticks to leave plenty of slack for slow CI.
    expected = BLOCK_SECONDS / HEARTBEAT_PERIOD
    assert ticks >= expected * 0.5, (
        f"event loop appears frozen: only {ticks} heartbeat ticks during a "
        f"{BLOCK_SECONDS}s sync tool call (expected ~{int(expected)})"
    )


@pytest.mark.asyncio
async def test_sync_tool_offload_preserves_return_value_and_kwargs():
    chat = Agentlys()

    def add(**kwargs) -> int:
        return kwargs["a"] + kwargs["b"]

    result = await chat._call_with_signature(add, None, a=2, b=40)
    assert result == 42


@pytest.mark.asyncio
async def test_async_tool_still_works():
    """Async tools must continue to be awaited inline, not offloaded."""
    chat = Agentlys()

    async def async_double(**kwargs) -> int:
        await asyncio.sleep(0)
        return kwargs["x"] * 2

    result = await chat._call_with_signature(async_double, None, x=21)
    assert result == 42


@pytest.mark.asyncio
async def test_from_response_is_passed_to_sync_tools():
    chat = Agentlys()

    captured = {}

    def tool(*, from_response):
        captured["from_response"] = from_response
        return "ok"

    sentinel = object()
    result = await chat._call_with_signature(tool, sentinel)
    assert result == "ok"
    assert captured["from_response"] is sentinel
