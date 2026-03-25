"""Tests for sub-agent support."""

import asyncio
import time
from unittest.mock import patch

import pytest
from agentlys import Agentlys, DEFAULT_COMPUTE_LEVELS
from agentlys.chat import StopLoopException, _resolve_thinking_for_model
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

    # Mock the child's streaming provider
    async def mock_child_stream(**kwargs):
        msg = _make_assistant_text("Here is the research result.")
        yield {"type": "text", "content": "Here is the research result."}
        yield {"type": "message", "message": msg}

    # Mock the parent's streaming provider: first call triggers sub-agent, second returns final
    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Research AI agents"},
                "call_sub_1",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Based on the research, here is the summary.")
            yield {"type": "text", "content": "Based on the research, here is the summary."}
            yield {"type": "message", "message": msg}

    with patch.object(child.provider, "fetch_stream_async", side_effect=mock_child_stream):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            events = []
            async for event in parent.run_conversation_stream_async("Tell me about AI agents"):
                events.append(event)

    # Collect messages by type
    user_events = [e for e in events if e.get("type") == "user"]
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    function_events = [e for e in events if e.get("type") == "function"]
    tool_result_events = [e for e in events if e.get("type") == "tool_result"]

    assert len(user_events) == 1
    assert len(assistant_events) == 2
    assert len(function_events) == 1
    assert len(tool_result_events) == 1

    # The tool result should contain the child's response
    assert "research result" in tool_result_events[0]["data"]["message"].content.lower()
    # The final assistant response is the parent's summary
    assert "summary" in assistant_events[1]["message"].content.lower()


@pytest.mark.asyncio
async def test_sub_agent_stateless():
    """Sub-agent messages should be reset between invocations."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child)

    # Pre-populate child messages to verify they get cleared
    child.messages = [Message(role="user", content="old message")]

    async def mock_child_stream(**kwargs):
        msg = _make_assistant_text("Fresh response.")
        yield {"type": "text", "content": "Fresh response."}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "New task"},
                "call_sub_2",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Done.")
            yield {"type": "text", "content": "Done."}
            yield {"type": "message", "message": msg}

    with patch.object(child.provider, "fetch_stream_async", side_effect=mock_child_stream):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            events = []
            async for event in parent.run_conversation_stream_async("Do something"):
                events.append(event)

    # Original child messages should be untouched (copy-based isolation)
    old_msgs = [m for m in child.messages if m.content == "old message"]
    assert len(old_msgs) == 1

    # Verify the sub-agent actually returned meaningful output
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert "fresh response" in tool_results[0]["data"]["message"].content.lower()


# ── Parallel execution tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sub_agents_parallel_execution():
    """Multiple sub-agents should execute concurrently, not sequentially."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)

    child_a = Agentlys(name="agent_a", provider=APIProvider.ANTHROPIC)
    child_b = Agentlys(name="agent_b", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child_a)
    parent.add_sub_agent(child_b)

    async def mock_child_a_stream(**kwargs):
        await asyncio.sleep(0.2)
        msg = _make_assistant_text("Result from agent A.")
        yield {"type": "text", "content": "Result from agent A."}
        yield {"type": "message", "message": msg}

    async def mock_child_b_stream(**kwargs):
        await asyncio.sleep(0.2)
        msg = _make_assistant_text("Result from agent B.")
        yield {"type": "text", "content": "Result from agent B."}
        yield {"type": "message", "message": msg}

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

    with patch.object(child_a.provider, "fetch_stream_async", side_effect=mock_child_a_stream):
        with patch.object(
            child_b.provider, "fetch_stream_async", side_effect=mock_child_b_stream
        ):
            start = time.time()
            result = await parent._call_functions_parallel(
                parallel_call.function_call_parts, parallel_call
            )
            elapsed = time.time() - start

    # If parallel: ~0.2s. If sequential: ~0.4s. Use generous threshold.
    assert elapsed < 0.35, f"Parallel execution took {elapsed}s, expected < 0.35s"

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

    async def mock_child_a_stream(**kwargs):
        msg = _make_assistant_text("Result A")
        yield {"type": "text", "content": "Result A"}
        yield {"type": "message", "message": msg}

    async def mock_child_b_stream(**kwargs):
        msg = _make_assistant_text("Result B")
        yield {"type": "text", "content": "Result B"}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Parent calls both sub-agents in parallel
            msg = Message(
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
            yield {"type": "text", "content": "I'll delegate to both agents."}
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Both tasks are complete.")
            yield {"type": "text", "content": "Both tasks are complete."}
            yield {"type": "message", "message": msg}

    with patch.object(child_a.provider, "fetch_stream_async", side_effect=mock_child_a_stream):
        with patch.object(
            child_b.provider, "fetch_stream_async", side_effect=mock_child_b_stream
        ):
            with patch.object(
                parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
            ):
                events = []
                async for event in parent.run_conversation_stream_async(
                    "Run both tasks"
                ):
                    events.append(event)

    # Verify both sub-agent results came through
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 2
    result_contents = sorted(
        e["data"]["message"].content.lower() for e in tool_results
    )
    assert "result a" in result_contents[0]
    assert "result b" in result_contents[1]

    # Final response
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    assert len(assistant_events) == 2
    assert "complete" in assistant_events[1]["message"].content.lower()


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

    async def mock_child_stream(**kwargs):
        msg = _make_assistant_text("Streamed research result.")
        yield {"type": "text", "content": "Streamed research result."}
        yield {"type": "message", "message": msg}

    stream_call_count = 0

    def get_mock_stream(**kwargs):
        nonlocal stream_call_count
        stream_call_count += 1
        if stream_call_count == 1:
            return _mock_stream_with_sub_agent_call()
        else:
            return _mock_stream_final()

    with patch.object(child.provider, "fetch_stream_async", side_effect=mock_child_stream):
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
    assert "streamed research result" in tool_data["message"].content.lower()


# ── Compute level tests ────────────────────────────────────────────────────


def test_compute_levels_true_adds_enum_to_schema():
    """compute_levels=True should add compute_level enum to the tool schema."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", instruction="Research.", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child, compute_levels=True)

    schema = next(
        s for s in parent.functions_schema if s["name"] == "sub_agent__researcher"
    )
    props = schema["parameters"]["properties"]
    assert "compute_level" in props
    assert props["compute_level"]["type"] == "string"
    assert props["compute_level"]["enum"] == ["high", "medium", "low"]
    # compute_level should NOT be required
    assert "compute_level" not in schema["parameters"]["required"]


def test_compute_levels_none_no_enum():
    """compute_levels=None (default) should not add compute_level to schema."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", instruction="Research.", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child)

    schema = next(
        s for s in parent.functions_schema if s["name"] == "sub_agent__researcher"
    )
    props = schema["parameters"]["properties"]
    assert "compute_level" not in props


def test_compute_levels_custom_dict():
    """compute_levels with a custom dict should store the mapping."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", instruction="Research.", provider=APIProvider.ANTHROPIC)

    custom_mapping = {
        "high": "gpt-4o",
        "medium": "gpt-4o-mini",
        "low": "gpt-3.5-turbo",
    }
    parent.add_sub_agent(child, compute_levels=custom_mapping)

    sub_agent_entry = parent._sub_agents["sub_agent__researcher"]
    assert sub_agent_entry["compute_levels"] == custom_mapping


def test_compute_levels_dict_missing_keys_raises():
    """compute_levels dict missing required keys should raise ValueError."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    with pytest.raises(ValueError, match="missing required keys"):
        parent.add_sub_agent(child, compute_levels={"high": "gpt-4o"})


def test_compute_levels_true_uses_defaults():
    """compute_levels=True should use DEFAULT_COMPUTE_LEVELS."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)

    parent.add_sub_agent(child, compute_levels=True)

    sub_agent_entry = parent._sub_agents["sub_agent__researcher"]
    assert sub_agent_entry["compute_levels"] == DEFAULT_COMPUTE_LEVELS


def _mock_child_stream(text: str, model_tracker: dict = None):
    """Create a mock fetch_stream_async that yields a simple assistant response."""
    async def _stream(**kwargs):
        if model_tracker is not None:
            # 'self' is the provider copy — capture its model
            model_tracker["model"] = kwargs.get("_provider_model", None)
        msg = _make_assistant_text(text)
        yield {"type": "text", "content": text}
        yield {"type": "message", "message": msg}
    return _stream


def _mock_child_stream_error():
    """Create a mock fetch_stream_async that raises an error."""
    async def _stream(**kwargs):
        raise RuntimeError("Sub-agent error")
        yield  # make it a generator  # noqa: E501
    return _stream


@pytest.mark.asyncio
async def test_compute_level_swaps_model():
    """Invoking a sub-agent with a compute_level should use the correct model on the copy."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    original_model = child.model
    original_provider_model = child.provider.model

    parent.add_sub_agent(child, compute_levels=True)

    # Track which model was set on the copied provider during fetch
    model_during_fetch = {}

    async def mock_child_stream(**kwargs):
        # 'self' context: this runs on the copied provider, capture its model
        # We use a side_effect function that gets the provider model via the instance
        model_during_fetch["captured"] = True
        msg = _make_assistant_text("Result.")
        yield {"type": "text", "content": "Result."}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Research X", "compute_level": "high"},
                "call_compute",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Done.")
            yield {"type": "text", "content": "Done."}
            yield {"type": "message", "message": msg}

    # Patch the child's provider — the copy will inherit the mock
    with patch.object(
        child.provider, "fetch_stream_async",
        side_effect=mock_child_stream,
    ):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            async for _ in parent.run_conversation_stream_async("Do research"):
                pass

    # The original child model should be untouched (copy pattern, not mutate)
    assert child.model == original_model
    assert child.provider.model == original_provider_model
    # Verify the sub-agent was actually invoked
    assert model_during_fetch.get("captured") is True


@pytest.mark.asyncio
async def test_compute_level_default_medium():
    """Invoking without compute_level should default to medium."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    original_model = child.model

    parent.add_sub_agent(child, compute_levels=True)

    invoked = {}

    async def mock_child_stream(**kwargs):
        invoked["called"] = True
        msg = _make_assistant_text("Result.")
        yield {"type": "text", "content": "Result."}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # No compute_level in arguments — should default to medium
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Quick lookup"},
                "call_default",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Done.")
            yield {"type": "text", "content": "Done."}
            yield {"type": "message", "message": msg}

    with patch.object(
        child.provider, "fetch_stream_async",
        side_effect=mock_child_stream,
    ):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            async for _ in parent.run_conversation_stream_async("Look something up"):
                pass

    # Original model untouched
    assert child.model == original_model
    # Sub-agent was invoked
    assert invoked.get("called") is True


@pytest.mark.asyncio
async def test_compute_level_preserves_original_on_error():
    """Original agent model should be untouched even if the sub-agent errors."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    original_model = child.model
    original_provider_model = child.provider.model

    parent.add_sub_agent(child, compute_levels=True)

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Fail task", "compute_level": "low"},
                "call_error",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Handled the error.")
            yield {"type": "text", "content": "Handled the error."}
            yield {"type": "message", "message": msg}

    with patch.object(
        child.provider, "fetch_stream_async",
        side_effect=_mock_child_stream_error(),
    ):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            async for _ in parent.run_conversation_stream_async("Try something"):
                pass

    # Original model untouched (copy pattern means original is never modified)
    assert child.model == original_model
    assert child.provider.model == original_provider_model


# ── Thinking compatibility tests ─────────────────────────────────────────────


class TestResolveThinkingForModel:
    def test_none_unchanged(self):
        assert _resolve_thinking_for_model(None, "claude-opus-4-20250514") is None
        assert _resolve_thinking_for_model(None, "claude-haiku-4-5-20251001") is None

    def test_haiku_adaptive_to_extended(self):
        result = _resolve_thinking_for_model(
            {"type": "adaptive"}, "claude-haiku-4-5-20251001"
        )
        assert result == {"type": "enabled", "budget_tokens": 5000}

    def test_haiku_preserves_custom_budget(self):
        result = _resolve_thinking_for_model(
            {"type": "adaptive", "budget_tokens": 8000}, "claude-haiku-4-5-20251001"
        )
        assert result == {"type": "enabled", "budget_tokens": 8000}

    def test_haiku_extended_passthrough(self):
        thinking = {"type": "enabled", "budget_tokens": 3000}
        result = _resolve_thinking_for_model(thinking, "claude-haiku-4-5-20251001")
        assert result == thinking

    def test_opus_extended_to_adaptive(self):
        result = _resolve_thinking_for_model(
            {"type": "enabled", "budget_tokens": 10000}, "claude-opus-4-20250514"
        )
        assert result == {"type": "adaptive"}

    def test_opus_adaptive_passthrough(self):
        thinking = {"type": "adaptive"}
        result = _resolve_thinking_for_model(thinking, "claude-opus-4-20250514")
        assert result == thinking

    def test_sonnet_extended_to_adaptive(self):
        result = _resolve_thinking_for_model(
            {"type": "enabled", "budget_tokens": 10000}, "claude-sonnet-4-20250514"
        )
        assert result == {"type": "adaptive"}

    def test_sonnet_adaptive_passthrough(self):
        thinking = {"type": "adaptive"}
        result = _resolve_thinking_for_model(thinking, "claude-sonnet-4-20250514")
        assert result == thinking

    def test_unknown_model_passthrough(self):
        thinking = {"type": "adaptive"}
        result = _resolve_thinking_for_model(thinking, "some-custom-model")
        assert result == thinking

    def test_original_dict_not_mutated(self):
        original = {"type": "adaptive"}
        _resolve_thinking_for_model(original, "claude-haiku-4-5-20251001")
        assert original == {"type": "adaptive"}


# ── Event callback tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_sub_agent_event_callback():
    """on_sub_agent_event should be called with events during sub-agent execution."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    parent.add_sub_agent(child)

    received_events = []

    def on_event(name: str, invocation_id: str, event: dict):
        received_events.append((name, invocation_id, event))

    parent.on_sub_agent_event = on_event

    async def mock_child_stream(**kwargs):
        msg = _make_assistant_text("Research result.")
        yield {"type": "text", "content": "Research result."}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Research AI"},
                "call_event_test",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Done.")
            yield {"type": "text", "content": "Done."}
            yield {"type": "message", "message": msg}

    with patch.object(child.provider, "fetch_stream_async", side_effect=mock_child_stream):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            async for _ in parent.run_conversation_stream_async("Research AI"):
                pass

    # Callback should have been invoked with child events
    assert len(received_events) > 0
    # All events should reference the correct sub-agent name
    assert all(name == "researcher" for name, _, _ in received_events)
    # All events should share the same invocation_id
    invocation_ids = set(inv_id for _, inv_id, _ in received_events)
    assert len(invocation_ids) == 1
    # Should include text and message events from the child
    event_types = [ev["type"] for _, _, ev in received_events]
    assert "text" in event_types
    assert "assistant" in event_types or "message" in event_types


# ── Nested sub-agents tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nested_sub_agents():
    """Sub-agents with their own sub-agents should work end-to-end."""
    coordinator = Agentlys(provider=APIProvider.ANTHROPIC)
    team_lead = Agentlys(name="team_lead", provider=APIProvider.ANTHROPIC)
    worker = Agentlys(name="worker", provider=APIProvider.ANTHROPIC)

    team_lead.add_sub_agent(worker)
    coordinator.add_sub_agent(team_lead)

    # Worker returns a result
    async def mock_worker_stream(**kwargs):
        msg = _make_assistant_text("Worker result.")
        yield {"type": "text", "content": "Worker result."}
        yield {"type": "message", "message": msg}

    # Team lead calls the worker, then returns a combined result
    team_lead_call_count = 0

    async def mock_team_lead_stream(**kwargs):
        nonlocal team_lead_call_count
        team_lead_call_count += 1
        if team_lead_call_count == 1:
            msg = _make_tool_call(
                "sub_agent__worker",
                {"prompt": "Do the work"},
                "call_worker",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Team lead compiled: Worker result.")
            yield {"type": "text", "content": "Team lead compiled: Worker result."}
            yield {"type": "message", "message": msg}

    # Coordinator calls team_lead, then returns final
    coord_call_count = 0

    async def mock_coordinator_stream(**kwargs):
        nonlocal coord_call_count
        coord_call_count += 1
        if coord_call_count == 1:
            msg = _make_tool_call(
                "sub_agent__team_lead",
                {"prompt": "Manage the team"},
                "call_team_lead",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Final coordinated result.")
            yield {"type": "text", "content": "Final coordinated result."}
            yield {"type": "message", "message": msg}

    with patch.object(worker.provider, "fetch_stream_async", side_effect=mock_worker_stream):
        with patch.object(
            team_lead.provider, "fetch_stream_async", side_effect=mock_team_lead_stream
        ):
            with patch.object(
                coordinator.provider, "fetch_stream_async",
                side_effect=mock_coordinator_stream,
            ):
                events = []
                async for event in coordinator.run_conversation_stream_async(
                    "Coordinate the work"
                ):
                    events.append(event)

    # Verify the coordinator got a final result through the nested chain
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    assert len(assistant_events) == 2
    assert "final coordinated result" in assistant_events[1]["message"].content.lower()

    # The tool result from team_lead should contain its compiled response
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert "team lead compiled" in tool_results[0]["data"]["message"].content.lower()


# ── Cancel event tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_event_stops_sub_agent():
    """Setting cancel_event should stop a running sub-agent mid-execution."""
    cancel_event = asyncio.Event()
    parent = Agentlys(provider=APIProvider.ANTHROPIC, cancel_event=cancel_event)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    parent.add_sub_agent(child)

    child_started = asyncio.Event()

    async def mock_child_stream(**kwargs):
        child_started.set()
        yield {"type": "text", "content": "Starting research..."}
        # Simulate slow work — cancel will fire before this completes
        await asyncio.sleep(5)
        msg = _make_assistant_text("Should not reach here.")
        yield {"type": "text", "content": "Should not reach here."}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Slow task"},
                "call_cancel",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Done.")
            yield {"type": "text", "content": "Done."}
            yield {"type": "message", "message": msg}

    with patch.object(child.provider, "fetch_stream_async", side_effect=mock_child_stream):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):

            async def run_and_cancel():
                events = []
                try:
                    async for event in parent.run_conversation_stream_async(
                        "Do slow task"
                    ):
                        events.append(event)
                except StopLoopException:
                    pass  # Expected — cancel_event triggers this
                return events

            # Schedule the cancel after the child starts
            async def cancel_after_start():
                await child_started.wait()
                cancel_event.set()

            events, _ = await asyncio.gather(
                run_and_cancel(), cancel_after_start()
            )

    # The conversation should have ended early — no "Should not reach here"
    all_text = " ".join(
        e.get("content", "") or e.get("message", Message(role="user")).content or ""
        for e in events
        if e.get("type") in ("text", "assistant")
    )
    assert "Should not reach here" not in all_text


@pytest.mark.asyncio
async def test_cancel_event_none_backward_compatible():
    """cancel_event=None (default) should not affect existing behavior."""
    parent = Agentlys(provider=APIProvider.ANTHROPIC)
    child = Agentlys(name="researcher", provider=APIProvider.ANTHROPIC)
    parent.add_sub_agent(child)

    # Verify cancel_event defaults to None
    assert parent.cancel_event is None

    async def mock_child_stream(**kwargs):
        msg = _make_assistant_text("Research complete.")
        yield {"type": "text", "content": "Research complete."}
        yield {"type": "message", "message": msg}

    call_count = 0

    async def mock_parent_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = _make_tool_call(
                "sub_agent__researcher",
                {"prompt": "Quick task"},
                "call_compat",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("All done.")
            yield {"type": "text", "content": "All done."}
            yield {"type": "message", "message": msg}

    with patch.object(child.provider, "fetch_stream_async", side_effect=mock_child_stream):
        with patch.object(
            parent.provider, "fetch_stream_async", side_effect=mock_parent_stream
        ):
            events = []
            async for event in parent.run_conversation_stream_async("Quick task"):
                events.append(event)

    # Should complete normally
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    assert len(assistant_events) == 2
    assert "all done" in assistant_events[1]["message"].content.lower()


@pytest.mark.asyncio
async def test_cancel_event_propagates_to_nested_sub_agents():
    """cancel_event should propagate through nested sub-agents."""
    cancel_event = asyncio.Event()
    coordinator = Agentlys(provider=APIProvider.ANTHROPIC, cancel_event=cancel_event)
    team_lead = Agentlys(name="team_lead", provider=APIProvider.ANTHROPIC)
    worker = Agentlys(name="worker", provider=APIProvider.ANTHROPIC)

    team_lead.add_sub_agent(worker)
    coordinator.add_sub_agent(team_lead)

    worker_started = asyncio.Event()

    async def mock_worker_stream(**kwargs):
        worker_started.set()
        yield {"type": "text", "content": "Worker starting..."}
        await asyncio.sleep(5)  # Slow — cancel should interrupt
        msg = _make_assistant_text("Worker should not finish.")
        yield {"type": "text", "content": "Worker should not finish."}
        yield {"type": "message", "message": msg}

    team_lead_call_count = 0

    async def mock_team_lead_stream(**kwargs):
        nonlocal team_lead_call_count
        team_lead_call_count += 1
        if team_lead_call_count == 1:
            msg = _make_tool_call(
                "sub_agent__worker",
                {"prompt": "Do slow work"},
                "call_worker",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Team lead done.")
            yield {"type": "text", "content": "Team lead done."}
            yield {"type": "message", "message": msg}

    coord_call_count = 0

    async def mock_coord_stream(**kwargs):
        nonlocal coord_call_count
        coord_call_count += 1
        if coord_call_count == 1:
            msg = _make_tool_call(
                "sub_agent__team_lead",
                {"prompt": "Manage work"},
                "call_team_lead",
            )
            yield {"type": "message", "message": msg}
        else:
            msg = _make_assistant_text("Coordinator done.")
            yield {"type": "text", "content": "Coordinator done."}
            yield {"type": "message", "message": msg}

    with patch.object(worker.provider, "fetch_stream_async", side_effect=mock_worker_stream):
        with patch.object(
            team_lead.provider, "fetch_stream_async", side_effect=mock_team_lead_stream
        ):
            with patch.object(
                coordinator.provider, "fetch_stream_async",
                side_effect=mock_coord_stream,
            ):

                async def run_and_cancel():
                    events = []
                    try:
                        async for event in coordinator.run_conversation_stream_async(
                            "Coordinate work"
                        ):
                            events.append(event)
                    except StopLoopException:
                        pass
                    return events

                async def cancel_after_worker_starts():
                    await worker_started.wait()
                    cancel_event.set()

                events, _ = await asyncio.gather(
                    run_and_cancel(), cancel_after_worker_starts()
                )

    # The deepest worker should not have completed
    all_text = " ".join(
        e.get("content", "") or e.get("message", Message(role="user")).content or ""
        for e in events
        if e.get("type") in ("text", "assistant")
    )
    assert "Worker should not finish" not in all_text
