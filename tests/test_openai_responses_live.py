"""Live checks of the Responses provider against api.openai.com.

Skipped unless OPENAI_LIVE=1 (needs a funded OPENAI_API_KEY). Uses the
cheapest reasoning model at low effort: a full run costs well under a cent.
"""

import os

import pytest
from agentlys import Agentlys
from agentlys.compaction import TokenThresholdCompaction
from agentlys.model import Message
from PIL import Image as PILImage

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_LIVE") != "1", reason="set OPENAI_LIVE=1 to hit the API"
)

MODEL = os.getenv("OPENAI_LIVE_MODEL", "gpt-5-nano")


def _agent(**kwargs):
    agent = Agentlys(
        provider="openai_responses",
        model=MODEL,
        effort="low",
        instruction="You are a terse assistant. Always use the tools you are given.",
        **kwargs,
    )
    agent.provider.max_tokens = 1500
    return agent


def get_weather(city: str, unit: str = "C") -> str:
    """Get the current weather for a city.

    Args:
        city: the city name
        unit: temperature unit, C or F
    """
    return f"21 degrees {unit}, sunny in {city}"


def render_chart(title: str) -> PILImage.Image:
    """Render a chart and return it as an image.

    Args:
        title: chart title
    """
    image = PILImage.new("RGB", (8, 8), color=(200, 30, 30))
    image.format = "PNG"
    return image


@pytest.mark.asyncio
async def test_tool_loop_replays_reasoning_and_optional_args():
    agent = _agent()
    agent.add_function(get_weather)  # optional arg -> strict must be false
    messages = [
        m async for m in agent.run_conversation_async("Weather in Paris, in C?")
    ]
    assistant_turns = [m for m in messages if m.role == "assistant"]
    assert any(m.function_call_parts for m in assistant_turns)
    assert "21" in messages[-1].content
    # The first assistant turn carried a replayable reasoning item.
    first = assistant_turns[0]
    thinking = [p for p in first.parts if p.type == "thinking"]
    assert thinking and thinking[0].thinking_signature.startswith("rs_")
    assert "|" in thinking[0].thinking_signature
    assert messages[-1].usage["output_tokens"] > 0


@pytest.mark.asyncio
async def test_streaming_events_and_image_tool_result():
    agent = _agent()
    agent.add_function(render_chart)
    events = [
        e
        async for e in agent.run_conversation_stream_async(
            "Render a chart titled 'sales' then describe its main colour in one word."
        )
    ]
    types = {e["type"] for e in events}
    assert "tool_started" in types and "tool_delta" in types
    assert "text" in types
    final_text = "".join(e["content"] for e in events if e["type"] == "text")
    assert final_text.strip()
    # The image tool result went out as a content array and was accepted.
    assert any(
        m["message"].role == "function" and m["message"].image is not None
        for m in events
        if m["type"] == "function"
    )


@pytest.mark.asyncio
async def test_history_reload_with_keep_thinking():
    agent = _agent()
    agent.add_function(get_weather)
    [m async for m in agent.run_conversation_async("Weather in Rome?")]
    history = list(agent.messages)

    fresh = _agent()
    fresh.add_function(get_weather)
    fresh.load_messages(history, keep_thinking=True)
    answer = await fresh.ask_async("And the unit you used, in one word?")
    assert answer.content


@pytest.mark.asyncio
async def test_compaction_and_complete():
    agent = _agent(compaction=TokenThresholdCompaction(token_threshold=1))
    await agent.ask_async("Say hi.")
    chunks = [c async for c in agent.ask_stream_async("Say hi again.")]
    assert any(c["type"] == "compacting" for c in chunks)
    assert agent.messages[0].has_compaction


@pytest.mark.asyncio
async def test_tool_search_loads_deferred_tool():
    agent = _agent()
    agent.add_function(get_weather)
    agent.add_function(render_chart)
    agent.enable_tool_search()
    messages = [
        m
        async for m in agent.run_conversation_async(
            "Search your tools for a weather tool, then give me the weather in Oslo."
        )
    ]
    called = [
        p.function_call["name"]
        for m in messages
        if m.role == "assistant"
        for p in m.function_call_parts
    ]
    assert "tool_search" in called and "get_weather" in called


@pytest.mark.asyncio
async def test_chat_completions_provider_reasoning_effort():
    agent = Agentlys(provider="openai", model=MODEL, effort="low")
    agent.add_function(get_weather)
    messages = [m async for m in agent.run_conversation_async("Weather in Lima?")]
    assert "21" in messages[-1].content
    assert messages[-1].usage["output_tokens"] > 0


def test_sync_ask():
    agent = _agent()
    assert (
        agent.ask("Reply with the single word pong.")
        .content.strip()
        .lower()
        .startswith("pong")
    )


@pytest.mark.asyncio
async def test_thinking_only_history_from_other_provider_is_ignored():
    agent = _agent()
    agent.load_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
        keep_thinking=True,
    )
    answer = await agent.ask_async("Reply with the single word pong.")
    assert "pong" in answer.content.lower()
