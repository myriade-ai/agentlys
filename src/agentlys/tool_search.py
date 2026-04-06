"""Tool search for agentlys – on-demand tool discovery via deferred loading."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from agentlys.base import AgentlysBase

logger = logging.getLogger(__name__)


@dataclass
class ToolSearchConfig:
    """Configuration for tool search behavior.

    Args:
        always_loaded: Tool names that should never be deferred.
        search_fn: Custom async search function with signature
            ``async def search(query: str, catalog: list[dict]) -> list[str]``.
            If *None*, uses a default LLM-based search with a cheap model.
        search_model: Model to use for the default LLM-based search.
    """

    always_loaded: list[str] = field(default_factory=list)
    search_fn: Optional[Callable] = None
    search_model: str = "claude-haiku-4-5-20251001"


_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "this", "that", "not", "no", "if", "can", "do", "get", "set", "all",
    "has", "have", "will", "been", "its", "also", "into", "use", "using",
    "des", "les", "une", "un", "le", "la", "et", "ou", "de", "du", "en",
    "pour", "par", "sur", "dans", "avec", "est", "ce", "qui", "que",
})


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens, filtering stop words."""
    import re

    tokens = set(re.split(r"[^a-z0-9]+", text.lower())) - {""}
    return tokens - _STOP_WORDS


async def _default_search(
    query: str,
    catalog: list[dict],
    chat: AgentlysBase,
    model: str,
) -> list[str]:
    """Default search using keyword matching.

    Scores each tool by how many query keywords appear in its name,
    description, and argument names.  Returns the top matches (up to 5).
    No LLM call needed — fast, free, and reliable.
    """
    if not catalog:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[int, str]] = []
    for tool in catalog:
        # Build searchable text from name, description, and args
        name = tool.get("name", "")
        desc = tool.get("description", "")
        args = tool.get("args", [])
        searchable = f"{name} {desc} {' '.join(args)}"
        tool_tokens = _tokenize(searchable)

        # Score = number of query tokens that match
        hits = len(query_tokens & tool_tokens)
        if hits > 0:
            scored.append((hits, name))

    # Require a minimum fraction of query tokens to match.
    # For 1-2 token queries, 1 hit suffices. For longer queries,
    # at least 30% of tokens must match to filter out noise.
    min_hits = max(1, int(len(query_tokens) * 0.3))
    scored = [(hits, name) for hits, name in scored if hits >= min_hits]

    # Sort by score descending, take top 5
    scored.sort(key=lambda x: x[0], reverse=True)
    matched = [name for _, name in scored[:5]]

    logger.info("Tool search query=%r matched=%s", query, matched)
    return matched


def create_search_tool_fn(
    chat: AgentlysBase,
) -> tuple[Callable, dict]:
    """Create the tool search function and its schema.

    Returns:
        A tuple of (async_callable, function_schema_dict).
    """
    config = chat._tool_search_config

    async def tool_search(query: str) -> list[str]:
        """Search for relevant tools based on a query description."""
        # Build catalog from deferred schemas (name, description, arg names)
        catalog = []
        for s in chat.functions_schema:
            if not s.get("defer_loading"):
                continue
            params = s.get("parameters", {})
            arg_names = list(params.get("properties", {}).keys())
            catalog.append({
                "name": s["name"],
                "description": s.get("description", ""),
                "args": arg_names,
            })

        if config.search_fn is not None:
            return await config.search_fn(query, catalog)
        else:
            return await _default_search(query, catalog, chat, config.search_model)

    schema = {
        "name": "tool_search",
        "description": (
            "Search for relevant tools by describing what you need. "
            "Returns tool definitions that match your query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of what tools you're looking for.",
                }
            },
            "required": ["query"],
        },
    }

    return tool_search, schema
