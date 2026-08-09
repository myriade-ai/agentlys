"""Tool results that are hard to serialize must not crash the conversation turn.

Formatting runs after the per-tool try/except of _execute_single_tool, so these
tests exercise the error boundary around _format_callback_message: the turn
must survive and the model must receive a usable function_result.
"""

from decimal import Decimal

import pytest
from agentlys import Agentlys
from agentlys.model import Message, MessagePart


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    """Agentlys() constructs an OpenAI client eagerly. These tests never hit
    the wire — just make the constructor happy."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")


def get_price() -> dict:
    """Return a row with a json-unserializable Decimal amount."""
    return {"amount": Decimal("19.99"), "currency": "EUR"}


def get_rows() -> list:
    """Return rows with heterogeneous keys."""
    return [{"a": 1}, {"b": 2, "c": 3}]


def get_raw_bytes() -> bytes:
    """Return bytes that are not an image."""
    return b"definitely not an image"


class SessionHandle:
    def close(self):
        """Close the session."""


def get_arbitrary_object() -> SessionHandle:
    """Return an arbitrary object."""
    return SessionHandle()


def _function_call_part(name: str, call_id: str) -> MessagePart:
    return MessagePart(
        type="function_call",
        function_call={"name": name, "arguments": {}},
        function_call_id=call_id,
    )


async def _run_single_tool(function, call_id: str) -> Message:
    agent = Agentlys()
    agent.add_function(function)
    part = _function_call_part(function.__name__, call_id)
    response = Message(role="assistant", parts=[part])
    return await agent._call_functions_parallel([part], response)


@pytest.mark.asyncio
async def test_decimal_dict_result_survives_the_turn():
    result = await _run_single_tool(get_price, "call_decimal")

    assert result.role == "function"
    assert result.parts[0].type == "function_result"
    assert "19.99" in result.parts[0].content


@pytest.mark.asyncio
async def test_heterogeneous_dict_list_result_survives_the_turn():
    result = await _run_single_tool(get_rows, "call_rows")

    assert result.role == "function"
    content = result.parts[0].content
    assert "a,b,c" in content  # header is the union of keys
    assert ",2,3" in content  # missing keys are filled with empty values


@pytest.mark.asyncio
async def test_arbitrary_object_result_survives_the_turn():
    result = await _run_single_tool(get_arbitrary_object, "call_object")

    assert result.role == "function"
    assert result.parts[0].type == "function_result"
    assert result.parts[0].content  # a usable, non-empty text result


@pytest.mark.asyncio
async def test_formatting_error_becomes_function_result():
    """Content that cannot be formatted (non-image bytes) is converted into an
    error function_result instead of raising through the whole turn."""
    result = await _run_single_tool(get_raw_bytes, "call_bytes")

    assert result.role == "function"
    assert result.parts[0].type == "function_result"
    assert result.parts[0].function_call_id == "call_bytes"
    assert "ValueError" in result.parts[0].content


@pytest.mark.asyncio
async def test_formatting_error_becomes_function_result_in_stream():
    agent = Agentlys()
    agent.add_function(get_raw_bytes)
    part = _function_call_part("get_raw_bytes", "call_bytes_stream")
    response = Message(role="assistant", parts=[part])

    results = []
    async for item in agent._stream_functions_parallel([part], response):
        results.append(item)

    assert len(results) == 1
    function_call_id, function_name, message = results[0]
    assert function_call_id == "call_bytes_stream"
    assert function_name == "get_raw_bytes"
    assert message.parts[0].type == "function_result"
    assert message.parts[0].function_call_id == "call_bytes_stream"
    assert "ValueError" in message.parts[0].content
