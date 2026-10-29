"""Tool results that are hard to serialize must not crash the conversation turn.

Formatting runs after the per-tool try/except of _execute_single_tool, so these
tests exercise the error boundary around _format_callback_message: the turn
must survive and the model must receive a usable function_result.
"""

import json
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import pytest
from agentlys import Agentlys
from agentlys.chat import OUTPUT_SIZE_LIMIT
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


# A cut the model cannot see is a cut it cannot work around: these pin the one
# marker every truncating path now shares.

OVERFLOW = OUTPUT_SIZE_LIMIT + 30_000


def get_long_text() -> str:
    """Return a string well past the output limit."""
    return "x" * OVERFLOW


def get_long_dict() -> dict:
    """Return a dict whose serialization is well past the output limit."""
    return {"rows": ["y" * 100 for _ in range(OVERFLOW // 100)]}


def get_long_string_list() -> list:
    """Return a list of strings well past the output limit."""
    return ["z" * 1000 for _ in range(OVERFLOW // 1000)]


@pytest.mark.parametrize(
    "function", [get_long_text, get_long_dict, get_long_string_list]
)
@pytest.mark.asyncio
async def test_an_oversized_result_says_how_much_it_lost(function):
    result = await _run_single_tool(function, f"call_{function.__name__}")
    content = result.parts[0].content

    assert "[truncated: " in content
    assert f"{OUTPUT_SIZE_LIMIT} of " in content
    assert "% dropped" in content
    # The advice matters more than the number: retrying returns the same cut.
    assert "Narrow the call" in content
    # The body itself is still bounded by the limit.
    marker_start = content.index("\n[truncated: ")
    assert marker_start == OUTPUT_SIZE_LIMIT


@pytest.mark.asyncio
async def test_a_result_that_fits_carries_no_marker():
    result = await _run_single_tool(get_price, "call_small")
    assert "[truncated: " not in result.parts[0].content


@pytest.mark.asyncio
async def test_a_second_cut_still_names_the_size_the_tool_produced():
    """An MCP result is capped by its server budget before the loop sees it.

    Reporting the intermediate length would tell the model 80% was dropped
    when 96% was — a confident lie about what it is missing.
    """
    from agentlys.utils import truncate_with_marker

    once = truncate_with_marker("D" * 500_000, 100_000)
    twice = truncate_with_marker(once, OUTPUT_SIZE_LIMIT)

    assert f"[truncated: {OUTPUT_SIZE_LIMIT} of 500000 characters shown" in twice
    assert "96% dropped" in twice


# A dict result is serialized only as far as the model will read it, the rest
# being measured for the marker (see test_event_loop_stall.py).  What the model
# receives must not depend on that: it is the text of the result dumped whole,
# then cut.


def _returning(value):
    def get_result():
        """Return the result."""
        return value

    return get_result


def _dict_dumping_to(length: int) -> dict:
    return {"k": "x" * (length - len(json.dumps({"k": ""})))}


def _circular() -> dict:
    loop = {}
    loop["itself"] = loop
    return loop


DICT_RESULTS = {
    "empty": {},
    "small": {"amount": Decimal("19.99"), "currency": "EUR"},
    "values json does not know": {
        "at": datetime(2026, 1, 2, 3, 4, 5),
        "id": UUID(int=7),
        "amount": Decimal("1.50"),
    },
    "keys that are not strings": {2: "int", 2.5: "float", True: "bool", None: "none"},
    "nested": {
        "rows": [
            {"id": i, "tags": ["a", "b"], "meta": {"ok": True, "n": None}}
            for i in range(3)
        ]
    },
    "not ascii": {"ville": "Besançon", "ok": "✅", "quote": 'she said "hi"\n'},
    "exactly the limit": _dict_dumping_to(OUTPUT_SIZE_LIMIT),
    "one character over": _dict_dumping_to(OUTPUT_SIZE_LIMIT + 1),
    "many small values, far over": {
        f"row_{i}": {"id": i, "label": "x" * 80, "amount": Decimal(i)}
        for i in range(2_000)
    },
    "one large value, far over": {
        "rows": [{"id": i, "label": "y" * 100} for i in range(OVERFLOW // 100)],
        "row_count": OVERFLOW // 100,
    },
}


@pytest.mark.parametrize("value", DICT_RESULTS.values(), ids=DICT_RESULTS.keys())
@pytest.mark.asyncio
async def test_a_dict_result_reads_as_if_dumped_whole_then_cut(value):
    from agentlys.utils import truncate_with_marker

    result = await _run_single_tool(_returning(value), "call_dict")

    dumped_whole = json.dumps(value, default=str)
    assert result.parts[0].content == truncate_with_marker(
        dumped_whole, OUTPUT_SIZE_LIMIT
    )


@pytest.mark.parametrize(
    "value",
    [{"loop": _circular()}, {"shown": "x" * OVERFLOW, "loop": _circular()}],
    ids=["within the limit", "past the limit"],
)
@pytest.mark.asyncio
async def test_a_dict_that_cannot_be_serialized_becomes_an_error_result(value):
    result = await _run_single_tool(_returning(value), "call_circular")

    assert result.role == "function"
    assert result.parts[0].type == "function_result"
    assert result.parts[0].function_call_id == "call_circular"
    assert "Circular reference detected" in result.parts[0].content


@pytest.mark.asyncio
async def test_a_dict_that_cannot_be_serialized_is_an_error_without_its_image():
    from PIL import Image as PILImage

    result = await _run_single_tool(
        _returning(({"loop": _circular()}, PILImage.new("RGB", (8, 8)))), "call_both"
    )

    (part,) = result.parts
    assert part.type == "function_result"
    assert "Circular reference detected" in part.content
