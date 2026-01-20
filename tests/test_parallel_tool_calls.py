"""Tests for parallel tool call execution."""

import asyncio
from unittest.mock import patch

import pytest
from agentlys import Agentlys
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import APIProvider


# Define multiple tools that could be called in parallel
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def get_time(city: str) -> str:
    """Get the current time for a city."""
    time_data = {
        "paris": "14:00 CET",
        "london": "13:00 GMT",
        "tokyo": "22:00 JST",
    }
    return time_data.get(city.lower(), f"Time data not available for {city}")


def get_population(city: str) -> str:
    """Get the population of a city."""
    population_data = {
        "paris": "2.1 million",
        "london": "8.8 million",
        "tokyo": "13.9 million",
    }
    return population_data.get(
        city.lower(), f"Population data not available for {city}"
    )


def _create_parallel_tool_call_message():
    """Create a mock assistant message with multiple parallel tool calls."""
    return Message(
        role="assistant",
        parts=[
            MessagePart(
                type="text",
                content="I'll get the weather and time for Paris.",
            ),
            MessagePart(
                type="function_call",
                function_call={"name": "get_weather", "arguments": {"city": "paris"}},
                function_call_id="call_weather_123",
            ),
            MessagePart(
                type="function_call",
                function_call={"name": "get_time", "arguments": {"city": "paris"}},
                function_call_id="call_time_456",
            ),
        ],
    )


def _create_final_response_message():
    """Create a mock final response message."""
    return Message(
        role="assistant",
        content="The weather in Paris is Sunny, 22°C and the time is 14:00 CET.",
    )


@pytest.mark.asyncio
async def test_parallel_tool_calls_async():
    """Test that multiple tool calls are executed in parallel via run_conversation_async."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(get_weather)
    agent.add_function(get_time)

    call_count = 0

    async def mock_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _create_parallel_tool_call_message()
        else:
            return _create_final_response_message()

    with patch.object(agent.provider, "fetch_async", side_effect=mock_fetch):
        messages = []
        async for message in agent.run_conversation_async(
            "What's the weather and time in Paris?"
        ):
            messages.append(message)

    # Verify we got messages: user, assistant with tool calls, function results, final response
    assert len(messages) == 4

    # First is user message
    assert messages[0].role == "user"
    assert messages[0].content == "What's the weather and time in Paris?"

    # Second is assistant with parallel tool calls
    assert messages[1].role == "assistant"
    assert len(messages[1].function_call_parts) == 2

    # Third is combined function results
    assert messages[2].role == "function"
    # Should have 2 function result parts (one for each tool)
    assert len(messages[2].parts) == 2

    # Fourth is final response
    assert messages[3].role == "assistant"


async def _mock_stream_with_parallel_calls():
    """Mock async generator that yields a response with multiple tool calls."""
    yield {"type": "text", "content": "I'll get the weather and time for London."}
    yield {
        "type": "message",
        "message": Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="text",
                    content="I'll get the weather and time for London.",
                ),
                MessagePart(
                    type="function_call",
                    function_call={
                        "name": "get_weather",
                        "arguments": {"city": "london"},
                    },
                    function_call_id="call_weather_789",
                ),
                MessagePart(
                    type="function_call",
                    function_call={"name": "get_time", "arguments": {"city": "london"}},
                    function_call_id="call_time_012",
                ),
            ],
        ),
    }


async def _mock_stream_final_response():
    """Mock async generator for final response after parallel tool calls."""
    yield {
        "type": "text",
        "content": "The weather in London is Cloudy, 15°C and the time is 13:00 GMT.",
    }
    yield {
        "type": "message",
        "message": Message(
            role="assistant",
            content="The weather in London is Cloudy, 15°C and the time is 13:00 GMT.",
        ),
    }


@pytest.mark.asyncio
async def test_parallel_tool_calls_stream_async():
    """Test that parallel tool calls work with streaming."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(get_weather)
    agent.add_function(get_time)

    call_count = 0

    def get_mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_stream_with_parallel_calls()
        else:
            return _mock_stream_final_response()

    with patch.object(
        agent.provider, "fetch_stream_async", side_effect=get_mock_stream
    ):
        events = []
        async for event in agent.run_conversation_stream_async(
            "What's the weather and time in London?"
        ):
            events.append(event)

    # Verify we got various event types
    user_events = [e for e in events if e.get("type") == "user"]
    text_events = [e for e in events if e.get("type") == "text"]
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    function_events = [e for e in events if e.get("type") == "function"]

    assert len(user_events) == 1
    assert len(text_events) >= 2  # Text from both responses
    assert len(assistant_events) == 2  # One with tool calls, one final
    assert len(function_events) == 1  # Combined function results

    # Check that function results message has both results
    function_msg = function_events[0]["message"]
    assert len(function_msg.parts) == 2


async def slow_tool_a() -> str:
    """A slow tool for testing parallel execution timing."""
    await asyncio.sleep(0.1)
    return "Result A"


async def slow_tool_b() -> str:
    """Another slow tool for testing parallel execution timing."""
    await asyncio.sleep(0.1)
    return "Result B"


@pytest.mark.asyncio
async def test_parallel_execution_is_concurrent():
    """Test that parallel tools actually execute concurrently, not sequentially."""
    from agentlys.model import MessagePart

    agent = Agentlys()
    agent.add_function(slow_tool_a)
    agent.add_function(slow_tool_b)

    # Create mock function call parts
    part_a = MessagePart(
        type="function_call",
        function_call={"name": "slow_tool_a", "arguments": {}},
        function_call_id="call_a",
    )
    part_b = MessagePart(
        type="function_call",
        function_call={"name": "slow_tool_b", "arguments": {}},
        function_call_id="call_b",
    )

    mock_response = Message(role="assistant", parts=[part_a, part_b])

    # Time the parallel execution
    import time

    start = time.time()
    result = await agent._call_functions_parallel([part_a, part_b], mock_response)
    elapsed = time.time() - start

    # If executed in parallel, should take ~0.1s, not ~0.2s
    # Using 0.15s as threshold to allow for some overhead
    assert elapsed < 0.15, f"Parallel execution took {elapsed}s, expected < 0.15s"

    # Verify both results are in the message
    assert result.role == "function"
    assert len(result.parts) == 2


# =============================================================================
# End-to-end tests with VCR cassettes
# =============================================================================


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_tool_calls_e2e_async():
    """End-to-end test for parallel tool calls using run_conversation_async."""
    agent = Agentlys(
        instruction="You are a helpful assistant. When asked for multiple pieces of information, call all relevant tools in parallel in a single response.",
        provider=APIProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
    )
    agent.add_function(get_weather)
    agent.add_function(get_time)

    user_query = "What is the weather and time in Paris? Use both tools."

    messages = []
    async for message in agent.run_conversation_async(user_query):
        messages.append(message)

    # Should have: user query, assistant with tool calls, function results, final response
    assert len(messages) >= 3

    # Find messages with function calls
    assistant_with_tools = [
        m for m in messages if m.role == "assistant" and m.function_call_parts
    ]

    # At least one assistant message should have made tool calls
    assert len(assistant_with_tools) >= 1

    # Check if parallel tool calls were made (2 tools in one message)
    max_tools_in_one_message = max(
        len(m.function_call_parts) for m in assistant_with_tools
    )
    assert max_tools_in_one_message == 2, (
        f"Expected 2 parallel tool calls, got {max_tools_in_one_message}"
    )

    # Verify function results exist
    function_messages = [m for m in messages if m.role == "function"]
    assert len(function_messages) >= 1

    # The function message should have results for both tools
    total_function_parts = sum(len(m.parts) for m in function_messages)
    assert total_function_parts >= 2
