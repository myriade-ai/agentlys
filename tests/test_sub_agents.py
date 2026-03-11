"""Tests for sub-agent support."""

import asyncio
import time
from unittest.mock import patch

import pytest
from agentlys import Agentlys
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import APIProvider


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_assistant_text(text: str) -> Message:
    return Message(role="assistant", content=text)


def _make_tool_call(func_name: str, args: dict, call_id: str) -> Message:
    return Message(
        role="assistant",
        parts=[
            MessagePart(
                type="function_call",
                function_call={"name": func_name, "arguments": args},
                function_call_id=call_id,
            )
        ],
    )


# ── Registration tests ──────────────────────────────────────────────────────


def test_add_sub_agent_registers_function():
    """add_sub_agent should register a single function with correct schema."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", instruction="You research topics.", provider=APIProvider.ANTHROPIC)

    func_name = parent.add_sub_agent(child)

    assert func_name == "sub_agent__researcher"
    assert "sub_agent__researcher" in parent.functions
    assert any(
        s["name"] == "sub_agent__researcher" for s in parent.functions_schema
    )

    # Verify schema structure
    schema = next(
        s for s in parent.functions_schema if s["name"] == "sub_agent__researcher"
    )
    assert schema["description"] == "You research topics."
    assert "prompt" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["prompt"]


def test_add_sub_agent_custom_name_and_description():
    """Custom name and description should override agent defaults."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="generic", instruction="Generic instruction.", provider=APIProvider.ANTHROPIC)

    func_name = parent.add_sub_agent(
        child, name="specialist", description="A specialist agent."
    )

    assert func_name == "sub_agent__specialist"
    schema = next(
        s for s in parent.functions_schema if s["name"] == "sub_agent__specialist"
    )
    assert schema["description"] == "A specialist agent."


def test_add_sub_agent_no_name_raises():
    """Should raise ValueError if sub-agent has no name."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(provider=APIProvider.ANTHROPIC)  # no name

    with pytest.raises(ValueError, match="must have a name"):
        parent.add_sub_agent(child)


def test_add_sub_agent_duplicate_raises():
    """Should raise ValueError if sub-agent name is already registered."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child1 = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    child2 = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child1)
    with pytest.raises(ValueError, match="already registered"):
        parent.add_sub_agent(child2)


def test_remove_sub_agent():
    """remove_sub_agent should clean up functions and schema."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child)
    assert "sub_agent__researcher" in parent.functions

    parent.remove_sub_agent("researcher")

    assert "sub_agent__researcher" not in parent.functions
    assert "sub_agent__researcher" not in parent._sub_agents
    assert not any(
        s["name"] == "sub_agent__researcher" for s in parent.functions_schema
    )


def test_remove_sub_agent_not_found_raises():
    """Should raise ValueError if sub-agent name not found."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)

    with pytest.raises(ValueError, match="No sub-agent"):
        parent.remove_sub_agent("nonexistent")


def test_reset_clears_sub_agents():
    """reset() should clear all sub-agents."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    parent.add_sub_agent(child)

    parent.reset()

    assert parent._sub_agents == {}


# ── Invocation tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sub_agent_invocation():
    """Sub-agent should run its conversation loop and return final assistant text."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child)

    # Mock the child's provider to return a simple response
    async def mock_child_fetch(**kwargs):
        return _make_assistant_text("Here is the research result.")

    # Mock the parent's provider: first call triggers the sub-agent, second returns final
    call_count = 0

    async def mock_parent_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Research AI agents"},
                "call_sub_1",
            )
        else:
            return _make_assistant_text("Based on the research, here is the summary.")

    with patch.object(child.provider, "fetch_async", side_effect=mock_child_fetch):
        with patch.object(
            parent.provider, "fetch_async", side_effect=mock_parent_fetch
        ):
            messages = []
            async for message in parent.run_conversation_async("Tell me about AI agents"):
                messages.append(message)

    # Expected: user, assistant (tool call), function result, final assistant
    assert len(messages) == 4
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].role == "function"
    assert messages[3].role == "assistant"

    # The function result should contain the child's response
    assert "research result" in messages[2].content.lower()
    # The final response is the parent's summary
    assert "summary" in messages[3].content.lower()


@pytest.mark.asyncio
async def test_sub_agent_stateless():
    """Sub-agent messages should be reset between invocations."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child)

    # Pre-populate child messages to verify they get cleared
    child.messages = [Message(role="user", content="old message")]

    async def mock_child_fetch(**kwargs):
        return _make_assistant_text("Fresh response.")

    call_count = 0

    async def mock_parent_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "New task"},
                "call_sub_2",
            )
        else:
            return _make_assistant_text("Done.")

    with patch.object(child.provider, "fetch_async", side_effect=mock_child_fetch):
        with patch.object(
            parent.provider, "fetch_async", side_effect=mock_parent_fetch
        ):
            async for _ in parent.run_conversation_async("Do something"):
                pass

    # Child messages should have been reset (not contain "old message")
    old_msgs = [m for m in child.messages if m.content == "old message"]
    assert len(old_msgs) == 0


# ── Parallel execution tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sub_agents_parallel_execution():
    """Multiple sub-agents should execute concurrently, not sequentially."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)

    child_a = Agentlys(name="agent_a", provider=APIProvider.ANTHROPIC)
    child_b = Agentlys(name="agent_b", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child_a)
    parent.add_sub_agent(child_b)

    async def mock_child_a_fetch(**kwargs):
        await asyncio.sleep(0.1)
        return _make_assistant_text("Result from agent A.")

    async def mock_child_b_fetch(**kwargs):
        await asyncio.sleep(0.1)
        return _make_assistant_text("Result from agent B.")

    # Parent calls both sub-agents in parallel
    parallel_call = Message(
        role="assistant",
        parts=[
            MessagePart(
                type="function_call",
                function_call={
                    "name": "sub_agent__agent_a",
                    "arguments": {"prompt": "Task A"},
                },
                function_call_id="call_a",
            ),
            MessagePart(
                type="function_call",
                function_call={
                    "name": "sub_agent__agent_b",
                    "arguments": {"prompt": "Task B"},
                },
                function_call_id="call_b",
            ),
        ],
    )

    with patch.object(child_a.provider, "fetch_async", side_effect=mock_child_a_fetch):
        with patch.object(
            child_b.provider, "fetch_async", side_effect=mock_child_b_fetch
        ):
            start = time.time()
            result = await parent._call_functions_parallel(
                parallel_call.function_call_parts, parallel_call
            )
            elapsed = time.time() - start

    # If parallel: ~0.1s. If sequential: ~0.2s.
    assert elapsed < 0.15, f"Parallel execution took {elapsed}s, expected < 0.15s"

    # Both results should be in the combined message
    assert result.role == "function"
    assert len(result.parts) == 2


@pytest.mark.asyncio
async def test_sub_agents_parallel_in_conversation():
    """Full conversation test with parallel sub-agent calls."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)

    child_a = Agentlys(name="agent_a", provider=APIProvider.ANTHROPIC)
    child_b = Agentlys(name="agent_b", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child_a, description="Agent A handles task A")
    parent.add_sub_agent(child_b, description="Agent B handles task B")

    async def mock_child_a_fetch(**kwargs):
        return _make_assistant_text("Result A")

    async def mock_child_b_fetch(**kwargs):
        return _make_assistant_text("Result B")

    call_count = 0

    async def mock_parent_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Parent calls both sub-agents in parallel
            return Message(
                role="assistant",
                parts=[
                    MessagePart(
                        type="text",
                        content="I'll delegate to both agents.",
                    ),
                    MessagePart(
                        type="function_call",
                        function_call={
                            "name": "sub_agent__agent_a",
                            "arguments": {"prompt": "Do task A"},
                        },
                        function_call_id="call_a",
                    ),
                    MessagePart(
                        type="function_call",
                        function_call={
                            "name": "sub_agent__agent_b",
                            "arguments": {"prompt": "Do task B"},
                        },
                        function_call_id="call_b",
                    ),
                ],
            )
        else:
            return _make_assistant_text("Both tasks are complete.")

    with patch.object(child_a.provider, "fetch_async", side_effect=mock_child_a_fetch):
        with patch.object(
            child_b.provider, "fetch_async", side_effect=mock_child_b_fetch
        ):
            with patch.object(
                parent.provider, "fetch_async", side_effect=mock_parent_fetch
            ):
                messages = []
                async for message in parent.run_conversation_async(
                    "Run both tasks"
                ):
                    messages.append(message)

    # Expected: user, assistant (parallel tool calls), function results, final assistant
    assert len(messages) == 4

    # Function results should have both sub-agent responses
    func_msg = messages[2]
    assert func_msg.role == "function"
    assert len(func_msg.parts) == 2

    # Final response
    assert messages[3].role == "assistant"
    assert "complete" in messages[3].content.lower()


# ── Streaming tests ─────────────────────────────────────────────────────────


async def _mock_stream_with_sub_agent_call():
    """Mock stream that yields a sub-agent tool call."""
    yield {"type": "text", "content": "Let me delegate this."}
    yield {
        "type": "message",
        "message": _make_tool_call(
            "sub_agent__researcher",
            {"prompt": "Research topic X"},
            "call_stream_sub",
        ),
    }


async def _mock_stream_final():
    """Mock stream for the final response."""
    yield {"type": "text", "content": "Here is the final answer."}
    yield {
        "type": "message",
        "message": _make_assistant_text("Here is the final answer."),
    }


@pytest.mark.asyncio
async def test_sub_agent_streaming():
    """Sub-agent should work with run_conversation_stream_async."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    parent.add_sub_agent(child)

    async def mock_child_fetch(**kwargs):
        return _make_assistant_text("Streamed research result.")

    stream_call_count = 0

    def get_mock_stream(**kwargs):
        nonlocal stream_call_count
        stream_call_count += 1
        if stream_call_count == 1:
            return _mock_stream_with_sub_agent_call()
        else:
            return _mock_stream_final()

    with patch.object(child.provider, "fetch_async", side_effect=mock_child_fetch):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=get_mock_stream
        ):
            events = []
            async for event in parent.run_conversation_stream_async(
                "Research topic X"
            ):
                events.append(event)

    # Verify we got the expected event types
    user_events = [e for e in events if e.get("type") == "user"]
    text_events = [e for e in events if e.get("type") == "text"]
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    function_events = [e for e in events if e.get("type") == "function"]
    tool_result_events = [e for e in events if e.get("type") == "tool_result"]

    assert len(user_events) == 1
    assert len(text_events) >= 2
    assert len(assistant_events) == 2
    assert len(function_events) == 1
    assert len(tool_result_events) == 1

    # Tool result should contain the child's response
    tool_data = tool_result_events[0]["data"]
    assert tool_data["function_name"] == "sub_agent__researcher"
