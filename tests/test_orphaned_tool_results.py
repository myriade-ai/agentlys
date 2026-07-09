"""Reverse-direction guard for the tool_use / tool_result contract.

``add_empty_function_result`` guards the *forward* direction (a ``function_call``
with no ``function_result``). ``drop_orphaned_function_results`` guards the
*reverse*: a ``function_result`` whose ``function_call_id`` matches no
``function_call`` in the history. Both OpenAI and Anthropic reject such an
orphaned result ("unexpected tool_use_id found in tool_result blocks", HTTP
400), which permanently blocks a conversation whose persisted history was left
with a dangling result — a stream interrupted mid-tool-use, or an assistant
tool_use turn deleted / regenerated after the result row was saved.
"""

import pytest

from agentlys import Agentlys
from agentlys.model import Message, MessagePart
from agentlys.providers.utils import drop_orphaned_function_results


def _assistant_call(call_id):
    return Message(
        role="assistant",
        parts=[
            MessagePart(
                type="function_call",
                function_call={"name": "get_weather", "arguments": {}},
                function_call_id=call_id,
            )
        ],
    )


def _function_result(call_id, content="ok"):
    return Message(
        role="function",
        parts=[
            MessagePart(
                type="function_result", content=content, function_call_id=call_id
            )
        ],
    )


def _result_ids(messages):
    return {
        part.function_call_id
        for message in messages
        for part in message.parts
        if part.type in ("function_result", "function_result_image")
    }


def test_orphaned_result_message_is_dropped():
    """A ``function`` message whose only result references a missing call is
    removed entirely, leaving the valid pair and surrounding turns intact."""
    messages = [
        Message(role="user", content="hello"),
        _assistant_call("toolu_valid"),
        _function_result("toolu_valid"),
        _function_result("toolu_orphan"),  # no matching function_call
    ]

    sanitized = drop_orphaned_function_results(messages)

    assert _result_ids(sanitized) == {"toolu_valid"}
    assert [m.role for m in sanitized] == ["user", "assistant", "function"]


def test_orphaned_part_stripped_but_sibling_result_kept():
    """A parallel-tool ``function`` message keeps its valid result and drops
    only the orphaned part; the message itself survives."""
    messages = [
        _assistant_call("toolu_valid"),
        Message(
            role="function",
            parts=[
                MessagePart(
                    type="function_result", content="ok", function_call_id="toolu_valid"
                ),
                MessagePart(
                    type="function_result",
                    content="orphan",
                    function_call_id="toolu_orphan",
                ),
            ],
        ),
    ]

    sanitized = drop_orphaned_function_results(messages)

    assert _result_ids(sanitized) == {"toolu_valid"}
    assert len(sanitized) == 2
    assert len(sanitized[1].parts) == 1


def test_valid_history_passes_through_by_reference():
    """A well-formed history is returned untouched, same object references (no
    needless copies on the hot path)."""
    messages = [
        Message(role="user", content="hi"),
        _assistant_call("toolu_a"),
        _function_result("toolu_a"),
    ]

    sanitized = drop_orphaned_function_results(messages)

    assert sanitized == messages
    assert _result_ids(sanitized) == {"toolu_a"}


def test_does_not_mutate_shared_history():
    """Must build new ``Message`` objects rather than strip parts in place — the
    input list holds the same references as ``chat.messages`` and this runs on
    every request."""
    orphan_message = Message(
        role="function",
        parts=[
            MessagePart(
                type="function_result", content="ok", function_call_id="toolu_valid"
            ),
            MessagePart(
                type="function_result",
                content="orphan",
                function_call_id="toolu_orphan",
            ),
        ],
    )
    messages = [_assistant_call("toolu_valid"), orphan_message]

    drop_orphaned_function_results(messages)

    # The original message still carries both parts — nothing mutated in place.
    assert len(orphan_message.parts) == 2


def _orphan_history():
    return [
        Message(role="user", content="hi"),
        _assistant_call("toolu_valid"),
        _function_result("toolu_valid"),
        _function_result("toolu_orphan"),
    ]


def test_anthropic_pipeline_drops_orphaned_tool_result():
    """End to end: an orphaned result never reaches the Anthropic request."""
    agent = Agentlys(
        provider="anthropic", model="claude-3-7-sonnet-latest", api_key="test"
    )
    agent.messages = _orphan_history()

    messages, _tools, _kwargs = agent.provider._prepare_request_params()

    tool_result_ids = {
        block.get("tool_use_id")
        for message in messages
        for block in (
            message["content"] if isinstance(message["content"], list) else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }
    assert "toolu_orphan" not in tool_result_ids
    assert "toolu_valid" in tool_result_ids


def test_openai_pipeline_drops_orphaned_tool_result(monkeypatch):
    """End to end: an orphaned result never reaches the OpenAI request."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent = Agentlys(provider="openai", model="gpt-4o")
    agent.messages = _orphan_history()

    messages, _tools, _kwargs = agent.provider._prepare_request_params()

    tool_call_ids = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    assert "toolu_orphan" not in tool_call_ids
    assert "toolu_valid" in tool_call_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
