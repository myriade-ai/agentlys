"""Unit tests for the MCP client glue (no LLM calls).

Uses an in-process FastMCP server over memory streams, so the full
list_tools/call_tool round-trip is exercised without subprocesses.
"""

import pytest
from agentlys import Agentlys
from agentlys.mcp import (
    convert_tool_result,
    fetch_mcp_server_tools,
    sanitize_tool_name,
)
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent


def make_server() -> FastMCP:
    mcp = FastMCP("Test")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @mcp.tool()
    def boom() -> str:
        """Always fails"""
        raise ValueError("kaboom")

    @mcp.tool()
    def long_output(n: int) -> str:
        """Return n characters"""
        return "x" * n

    @mcp.tool(name="weird.name/check")
    def weird() -> str:
        """Tool with a name LLM providers reject"""
        return "ok"

    return mcp


def test_sanitize_tool_name():
    assert sanitize_tool_name("add") == "add"
    assert sanitize_tool_name("add", prefix="srv__") == "srv__add"
    assert sanitize_tool_name("weird.name/check") == "weird_name_check"
    assert len(sanitize_tool_name("x" * 100, prefix="p__")) == 64


def test_convert_tool_result_multiple_blocks():
    result = CallToolResult(
        content=[
            TextContent(type="text", text="part one"),
            TextContent(type="text", text="part two"),
        ]
    )
    assert convert_tool_result(result) == "part one\npart two"


def test_convert_tool_result_empty_and_structured():
    assert convert_tool_result(CallToolResult(content=[])) == "(empty result)"
    result = CallToolResult(content=[], structuredContent={"answer": 42})
    assert convert_tool_result(result) == '{"answer": 42}'


def test_convert_tool_result_error():
    result = CallToolResult(
        content=[TextContent(type="text", text="it broke")], isError=True
    )
    assert convert_tool_result(result) == "MCP tool error: it broke"


def test_convert_tool_result_truncation():
    result = CallToolResult(content=[TextContent(type="text", text="x" * 500)])
    converted = convert_tool_result(result, max_result_chars=100)
    assert converted.startswith("x" * 100)
    assert "truncated to 100 characters" in converted
    assert len(converted) < 250


@pytest.mark.asyncio
async def test_fetch_tools_prefix_sanitization_and_call():
    server = make_server()
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        functions, schemas = await fetch_mcp_server_tools(session, prefix="mcp_demo__")
        names = {schema["name"] for schema in schemas}
        assert "mcp_demo__add" in names
        assert "mcp_demo__weird_name_check" in names
        # only the keys agentlys understands survive (untrusted extras dropped)
        assert all(
            set(schema) == {"name", "description", "parameters"} for schema in schemas
        )
        # the wrapper calls the ORIGINAL (unprefixed) server-side name
        assert await functions["mcp_demo__add"](a=1, b=2) == "3"
        assert await functions["mcp_demo__weird_name_check"]() == "ok"


@pytest.mark.asyncio
async def test_fetch_tools_filter():
    server = make_server()
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        functions, schemas = await fetch_mcp_server_tools(
            session, tool_filter=lambda tool: tool.name == "add"
        )
        assert [schema["name"] for schema in schemas] == ["add"]
        assert list(functions) == ["add"]


@pytest.mark.asyncio
async def test_fetch_tools_error_result_does_not_raise():
    server = make_server()
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        functions, _ = await fetch_mcp_server_tools(session)
        result = await functions["boom"]()
        assert result.startswith("MCP tool error:")
        assert "kaboom" in result


@pytest.mark.asyncio
async def test_fetch_tools_result_cap():
    server = make_server()
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        functions, _ = await fetch_mcp_server_tools(session, max_result_chars=50)
        result = await functions["long_output"](n=5000)
        assert "truncated to 50 characters" in result
        assert len(result) < 200


@pytest.mark.asyncio
async def test_add_mcp_server_defers_with_tool_search(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    server = make_server()
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        agent = Agentlys(provider="anthropic")

        def dummy() -> str:
            """A built-in tool"""
            return "x"

        agent.add_function(dummy)
        agent.enable_tool_search(always_loaded=["dummy"])

        registered = await agent.add_mcp_server(
            session, prefix="srv__", include_resources=False
        )
        assert "srv__add" in registered
        schema = next(s for s in agent.functions_schema if s["name"] == "srv__add")
        # tools added after enable_tool_search are auto-deferred
        assert schema.get("defer_loading") is True
        assert agent.functions["srv__add"] is not None

        # re-registering the same server never duplicates nor shadows
        assert (
            await agent.add_mcp_server(session, prefix="srv__", include_resources=False)
            == []
        )
