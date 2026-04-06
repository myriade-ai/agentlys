"""Tests for tool search support."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentlys import Agentlys, APIProvider, ToolSearchConfig
from agentlys.model import Message, MessagePart
from agentlys.providers.anthropic import part_to_anthropic_dict
from agentlys.tool_search import create_search_tool_fn, _default_search


# ── Helpers ──────────────────────────────────────────────────────────────────


def _dummy_fn_a(query: str) -> str:
    """Search the web for information."""
    return f"result for {query}"


def _dummy_fn_b(city: str) -> str:
    """Get the current weather for a city."""
    return f"weather in {city}"


def _dummy_fn_c(path: str) -> str:
    """Read a file from disk."""
    return f"contents of {path}"


# ── Registration tests ───────────────────────────────────────────────────────


def test_add_function_defer_loading_flag():
    """add_function with defer_loading=True sets the flag on the schema."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a, defer_loading=True)

    schema = agent.functions_schema[0]
    assert schema["defer_loading"] is True
    assert schema["name"] == "_dummy_fn_a"
    assert "_dummy_fn_a" in agent.functions


def test_add_function_no_defer_loading_by_default():
    """add_function without defer_loading does not set the flag."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)

    schema = agent.functions_schema[0]
    assert "defer_loading" not in schema


def test_enable_tool_search_defers_existing_tools():
    """enable_tool_search marks all existing tools as deferred."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.add_function(_dummy_fn_b)

    agent.enable_tool_search()

    # Both existing tools should be deferred
    for schema in agent.functions_schema:
        if schema["name"] in ("_dummy_fn_a", "_dummy_fn_b"):
            assert schema.get("defer_loading") is True

    # The search tool itself should NOT be deferred
    search_schema = next(
        s for s in agent.functions_schema if s["name"] == "tool_search"
    )
    assert "defer_loading" not in search_schema or not search_schema.get(
        "defer_loading"
    )


def test_enable_tool_search_always_loaded():
    """Tools in always_loaded should not be deferred."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.add_function(_dummy_fn_b)

    agent.enable_tool_search(always_loaded=["_dummy_fn_a"])

    schema_a = next(
        s for s in agent.functions_schema if s["name"] == "_dummy_fn_a"
    )
    schema_b = next(
        s for s in agent.functions_schema if s["name"] == "_dummy_fn_b"
    )

    assert "defer_loading" not in schema_a or not schema_a.get("defer_loading")
    assert schema_b.get("defer_loading") is True


def test_auto_defer_tools_added_after_enable():
    """Tools added after enable_tool_search are automatically deferred."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.enable_tool_search()
    agent.add_function(_dummy_fn_c)

    schema = next(
        s for s in agent.functions_schema if s["name"] == "_dummy_fn_c"
    )
    assert schema.get("defer_loading") is True


def test_auto_defer_respects_always_loaded():
    """Tools in always_loaded are not auto-deferred even when added after enable."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.enable_tool_search(always_loaded=["_dummy_fn_c"])
    agent.add_function(_dummy_fn_c)

    schema = next(
        s for s in agent.functions_schema if s["name"] == "_dummy_fn_c"
    )
    assert "defer_loading" not in schema or not schema.get("defer_loading")


def test_search_tool_registered():
    """enable_tool_search registers a 'tool_search' function."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.enable_tool_search()

    assert "tool_search" in agent.functions
    assert any(s["name"] == "tool_search" for s in agent.functions_schema)
    assert agent._tool_search_function_name == "tool_search"


def test_reset_clears_tool_search_state():
    """reset() should clear all tool search state."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.enable_tool_search()

    agent.reset()

    assert agent._tool_search_config is None
    assert agent._tool_search_function_name is None


def test_tool_search_config_default_model_anthropic():
    """Default search model for Anthropic should be Haiku."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.enable_tool_search()

    assert agent._tool_search_config.search_model == "claude-haiku-4-5-20251001"


def test_tool_search_config_custom_model():
    """Custom search model should be used."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.enable_tool_search(search_model="claude-sonnet-4-5-20250514")

    assert agent._tool_search_config.search_model == "claude-sonnet-4-5-20250514"


# ── Provider formatting tests ────────────────────────────────────────────────


def test_anthropic_tool_reference_formatting():
    """part_to_anthropic_dict should format tool_references as tool_reference blocks."""
    part = MessagePart(
        type="function_result",
        content="get_weather, read_file",
        function_call_id="call_123",
        tool_references=["get_weather", "read_file"],
    )

    result = part_to_anthropic_dict(part)

    assert result == {
        "type": "tool_result",
        "tool_use_id": "call_123",
        "content": [
            {"type": "tool_reference", "tool_name": "get_weather"},
            {"type": "tool_reference", "tool_name": "read_file"},
        ],
    }


def test_anthropic_regular_function_result_unchanged():
    """part_to_anthropic_dict should handle regular function results unchanged."""
    part = MessagePart(
        type="function_result",
        content="some result text",
        function_call_id="call_456",
    )

    result = part_to_anthropic_dict(part)

    assert result == {
        "type": "tool_result",
        "tool_use_id": "call_456",
        "content": "some result text",
    }


def test_anthropic_tools_array_includes_defer_loading():
    """Anthropic provider should pass defer_loading flag in tools array."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a, defer_loading=True)
    agent.add_function(_dummy_fn_b)

    # Access the provider to build request params
    agent.messages = [Message(role="user", content="test")]
    messages, tools, kwargs = agent.provider._prepare_request_params()

    tool_a = next(t for t in tools if t["name"] == "_dummy_fn_a")
    tool_b = next(t for t in tools if t["name"] == "_dummy_fn_b")

    assert tool_a.get("defer_loading") is True
    assert "defer_loading" not in tool_b or not tool_b.get("defer_loading")


# ── format_callback_message tests ────────────────────────────────────────────


def test_format_callback_tool_search_result():
    """_format_callback_message should produce tool_references for search results."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.add_function(_dummy_fn_b)
    agent.enable_tool_search()

    result = agent._format_callback_message(
        function_name="tool_search",
        function_call_id="call_789",
        content=["_dummy_fn_a", "_dummy_fn_b"],
        image=None,
    )

    assert len(result.parts) == 1
    part = result.parts[0]
    assert part.type == "function_result"
    assert part.tool_references == ["_dummy_fn_a", "_dummy_fn_b"]
    assert part.function_call_id == "call_789"


def test_format_callback_empty_search_result():
    """_format_callback_message should handle empty search results."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.enable_tool_search()

    result = agent._format_callback_message(
        function_name="tool_search",
        function_call_id="call_000",
        content=[],
        image=None,
    )

    part = result.parts[0]
    assert part.tool_references == []
    assert part.content == "[]"


def test_format_callback_regular_function_unchanged():
    """_format_callback_message for regular functions should not set tool_references."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.enable_tool_search()

    result = agent._format_callback_message(
        function_name="_dummy_fn_a",
        function_call_id="call_111",
        content="some result",
        image=None,
    )

    part = result.parts[0]
    assert part.tool_references is None
    assert part.content == "some result"


# ── Default search function tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_search_empty_catalog():
    """Default search should return empty list for empty catalog."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    result = await _default_search("find weather tools", [], agent, "claude-haiku-4-5-20251001")
    assert result == []


@pytest.mark.asyncio
async def test_default_search_keyword_matching():
    """Default search should match query keywords against tool names and descriptions."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)

    catalog = [
        {"name": "get_weather", "description": "Get current weather for a location"},
        {"name": "read_file", "description": "Read a file from disk"},
        {"name": "EchartsTool-echarts__preview_render", "description": "Render a chart preview"},
    ]

    result = await _default_search("weather", catalog, agent, "claude-haiku-4-5-20251001")
    assert "get_weather" in result
    assert "read_file" not in result


@pytest.mark.asyncio
async def test_default_search_partial_keyword():
    """Default search should match partial keywords from tool names."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)

    catalog = [
        {"name": "EchartsTool-echarts__preview_render", "description": "Render a chart"},
        {"name": "EchartsTool-echarts__get_chart_configuration", "description": "Get config for chart type"},
        {"name": "CatalogTool-catalog__list_assets", "description": "List catalog assets"},
    ]

    # "render" should match preview_render
    result = await _default_search("render", catalog, agent, "claude-haiku-4-5-20251001")
    assert "EchartsTool-echarts__preview_render" in result

    # "chart" should match both echarts tools
    result = await _default_search("chart", catalog, agent, "claude-haiku-4-5-20251001")
    assert "EchartsTool-echarts__preview_render" in result
    assert "EchartsTool-echarts__get_chart_configuration" in result
    assert "CatalogTool-catalog__list_assets" not in result


@pytest.mark.asyncio
async def test_default_search_no_match():
    """Default search should return empty list when nothing matches."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)

    catalog = [
        {"name": "get_weather", "description": "Get weather"},
    ]

    result = await _default_search("completely unrelated xyz", catalog, agent, "claude-haiku-4-5-20251001")
    assert result == []


# ── Stop words and threshold tests ───────────────────────────────────────────

REALISTIC_CATALOG = [
    {"name": "EchartsTool-echarts__preview_render", "description": "Render a chart (using Echarts 6). This is not shown to the user, but this will create a preview."},
    {"name": "EchartsTool-echarts__get_chart_configuration", "description": "Get detailed configuration options for a specific chart type."},
    {"name": "EchartsTool-echarts__list_chart_types", "description": "List all available ECharts chart types with descriptions."},
    {"name": "CatalogTool-catalog__get_asset_activities", "description": "Get the full activity feed for an asset, including content and changes."},
    {"name": "CatalogTool-catalog__list_assets", "description": "List catalog assets and terms with optional filtering"},
    {"name": "CatalogTool-catalog__update_asset", "description": "Update catalog asset documentation. Descriptions are in Markdown."},
    {"name": "DocumentsTool-documents__create_document", "description": "Create a new report document with the given title and content."},
    {"name": "DocumentsTool-documents__list_documents", "description": "List all reports/documents for the current database."},
    {"name": "DocumentsTool-documents__search_documents", "description": "Search reports by title or content."},
    {"name": "DocumentEditor-document_editor__edit_document", "description": "Replace all occurrences of 'old_string' with 'new_string' in the document."},
    {"name": "DocumentEditor-document_editor__read_document", "description": "Read a document with line numbers."},
    {"name": "MemoryTool-memory__initialize_memory", "description": "Initialize memory sections for organization or user scope."},
]


@pytest.mark.asyncio
async def test_stop_words_filtered():
    """Stop words like 'and', 'for', 'with' should not cause matches."""
    result = await _default_search(
        "chart creation and rendering", REALISTIC_CATALOG, None, ""
    )
    # Should match EchartsTool only, not CatalogTool (via 'and')
    for name in result:
        assert "EchartsTool" in name or "chart" in name.lower()


@pytest.mark.asyncio
async def test_stop_words_french():
    """French stop words should also be filtered."""
    result = await _default_search(
        "les documents et les rapports", REALISTIC_CATALOG, None, ""
    )
    # 'les', 'et' are stop words — only 'documents' and 'rapports' should match
    assert any("documents" in name.lower() or "document" in name.lower() for name in result)
    # Should NOT match unrelated tools via 'les' or 'et'
    assert all(
        "document" in name.lower() for name in result
    )


@pytest.mark.asyncio
async def test_threshold_filters_weak_matches():
    """Multi-word queries should require more than 1 token to match."""
    result = await _default_search(
        "render chart configuration preview echarts", REALISTIC_CATALOG, None, ""
    )
    # 5 meaningful tokens -> min 2 hits needed
    # EchartsTool tools should match (multiple hits), CatalogTool should not
    for name in result:
        assert "EchartsTool" in name


@pytest.mark.asyncio
async def test_single_word_query_still_works():
    """A single word query should still return matches with 1 hit."""
    result = await _default_search("render", REALISTIC_CATALOG, None, "")
    assert "EchartsTool-echarts__preview_render" in result


@pytest.mark.asyncio
async def test_query_only_stop_words():
    """A query made entirely of stop words should return no results."""
    result = await _default_search("and or the for with", REALISTIC_CATALOG, None, "")
    assert result == []


@pytest.mark.asyncio
async def test_document_tools_match():
    """Document-related queries should find document tools ranked first."""
    result = await _default_search("create document", REALISTIC_CATALOG, None, "")
    assert "DocumentsTool-documents__create_document" in result
    # create_document should rank highest (2 hits: create + document)
    assert result[0] == "DocumentsTool-documents__create_document"


@pytest.mark.asyncio
async def test_edit_document_query():
    """Editing queries should match document editor tools."""
    result = await _default_search("edit document content", REALISTIC_CATALOG, None, "")
    assert "DocumentEditor-document_editor__edit_document" in result


@pytest.mark.asyncio
async def test_asset_query_matches_catalog():
    """Asset-related queries should match catalog tools."""
    result = await _default_search("update asset documentation", REALISTIC_CATALOG, None, "")
    assert "CatalogTool-catalog__update_asset" in result


@pytest.mark.asyncio
async def test_memory_query():
    """Memory queries should match memory tools."""
    result = await _default_search("initialize memory", REALISTIC_CATALOG, None, "")
    assert "MemoryTool-memory__initialize_memory" in result
    assert len(result) == 1


@pytest.mark.asyncio
async def test_results_ranked_by_score():
    """Tools with more matching tokens should rank higher."""
    result = await _default_search(
        "list chart types echarts", REALISTIC_CATALOG, None, ""
    )
    # list_chart_types should rank first (matches: list, chart, types, echarts)
    assert result[0] == "EchartsTool-echarts__list_chart_types"


@pytest.mark.asyncio
async def test_max_5_results():
    """Should return at most 5 results."""
    # Query that could match many tools
    result = await _default_search("document", REALISTIC_CATALOG, None, "")
    assert len(result) <= 5


# ── Custom search function tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_search_fn():
    """enable_tool_search with custom search_fn should use it."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)
    agent.add_function(_dummy_fn_b)

    async def my_search(query: str, catalog: list[dict]) -> list[str]:
        # Always return the first tool
        return [catalog[0]["name"]] if catalog else []

    agent.enable_tool_search(search_fn=my_search)

    # Get the search tool and call it
    search_fn = agent.functions["tool_search"]
    result = await search_fn(query="anything")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "_dummy_fn_a"


# ── create_search_tool_fn tests ──────────────────────────────────────────────


def test_create_search_tool_fn_schema():
    """create_search_tool_fn should return a valid schema."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent._tool_search_config = ToolSearchConfig()

    fn, schema = create_search_tool_fn(agent)

    assert schema["name"] == "tool_search"
    assert "query" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["query"]
    assert callable(fn)


# ── ToolSearchConfig tests ───────────────────────────────────────────────────


def test_tool_search_config_defaults():
    """ToolSearchConfig should have sensible defaults."""
    config = ToolSearchConfig()

    assert config.always_loaded == []
    assert config.search_fn is None
    assert config.search_model == "claude-haiku-4-5-20251001"


def test_tool_search_config_custom():
    """ToolSearchConfig should accept custom values."""
    async def custom_search(query, catalog):
        return []

    config = ToolSearchConfig(
        always_loaded=["tool_a"],
        search_fn=custom_search,
        search_model="custom-model",
    )

    assert config.always_loaded == ["tool_a"]
    assert config.search_fn is custom_search
    assert config.search_model == "custom-model"


# ── Idempotency tests ────────────────────────────────────────────────────────


def test_enable_tool_search_twice_no_duplicates():
    """Calling enable_tool_search twice should not register duplicate search tools."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)

    agent.enable_tool_search()
    agent.enable_tool_search()

    search_schemas = [
        s for s in agent.functions_schema if s["name"] == "tool_search"
    ]
    assert len(search_schemas) == 1


def test_enable_tool_search_twice_updates_config():
    """Calling enable_tool_search twice should update the config."""
    agent = Agentlys(provider=APIProvider.ANTHROPIC)
    agent.add_function(_dummy_fn_a)

    agent.enable_tool_search(always_loaded=["_dummy_fn_a"])
    assert agent._tool_search_config.always_loaded == ["_dummy_fn_a"]

    agent.enable_tool_search(always_loaded=[])
    assert agent._tool_search_config.always_loaded == []


# ── Anthropic empty tool_references test ─────────────────────────────────────


def test_anthropic_empty_tool_references():
    """Empty tool_references should produce an empty tool_reference array."""
    part = MessagePart(
        type="function_result",
        content="[]",
        function_call_id="call_empty",
        tool_references=[],
    )

    result = part_to_anthropic_dict(part)

    assert result == {
        "type": "tool_result",
        "tool_use_id": "call_empty",
        "content": [],
    }
