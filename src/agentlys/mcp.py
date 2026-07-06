import json
import logging
import re
import typing

from mcp import ClientSession

logger = logging.getLogger(__name__)

# Anthropic and OpenAI both constrain tool names to this charset and length.
TOOL_NAME_MAX_LENGTH = 64
_TOOL_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")

# Filter signature: receives the MCP Tool object, returns True to expose it.
ToolFilter = typing.Callable[[typing.Any], bool]


def _truncate_result(
    text: str, max_result_chars: typing.Optional[int], hint: str = ""
) -> str:
    """Truncate an MCP result to max_result_chars with an explicit marker."""
    if max_result_chars is None or len(text) <= max_result_chars:
        return text
    return (
        text[:max_result_chars]
        + f"\n[... truncated to {max_result_chars} characters.{hint}]"
    )


def sanitize_tool_name(name: str, prefix: str = "") -> str:
    """Make an MCP tool name safe for LLM providers.

    Applies the optional namespace prefix, replaces characters outside
    ``[a-zA-Z0-9_-]`` with ``_`` and truncates to the provider limit.
    """
    sanitized = _TOOL_NAME_SANITIZE_RE.sub("_", f"{prefix}{name}")
    return sanitized[:TOOL_NAME_MAX_LENGTH]


def convert_tool_result(result, max_result_chars: typing.Optional[int] = None) -> str:
    """Convert an MCP ``CallToolResult`` into a string for the LLM.

    Handles multiple content blocks, non-text blocks, structured content,
    ``isError`` results and optional truncation. Never raises on the result
    shape so a misbehaving server cannot crash the agent loop.
    """
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        block_type = getattr(block, "type", "unknown")
        if block_type == "text":
            parts.append(block.text or "")
        else:
            parts.append(f"[{block_type} content omitted]")

    text = "\n".join(part for part in parts if part)

    if not text:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                text = json.dumps(structured, default=str)
            except (TypeError, ValueError):
                text = str(structured)

    if getattr(result, "isError", False):
        text = f"MCP tool error: {text or 'unknown error'}"

    if not text:
        text = "(empty result)"

    return _truncate_result(
        text,
        max_result_chars,
        hint=" Refine the call (filters, pagination) to get less data.",
    )


async def fetch_mcp_server_tools(
    mcp_server: ClientSession,
    prefix: str = "",
    tool_filter: typing.Optional[ToolFilter] = None,
    max_result_chars: typing.Optional[int] = None,
):
    """
    Fetch the tools from the MCP server and return (functions, schemas).

    Args:
        mcp_server: An initialized MCP ``ClientSession``.
        prefix: Namespace prefix applied to every tool name (collision
            protection when multiple servers — or built-in tools — coexist).
        tool_filter: Optional callable receiving the MCP ``Tool`` object;
            return False to skip the tool (host-side allowlisting / pinning).
        max_result_chars: Optional cap on the size of tool results.
    """
    schemas = []
    functions = {}
    tools = (await mcp_server.list_tools()).tools

    for tool in tools:
        if tool_filter is not None and not tool_filter(tool):
            continue

        name = sanitize_tool_name(tool.name, prefix)
        if name in functions:
            logger.warning("Duplicate MCP tool name %r, skipping", name)
            continue

        # Only keep the keys agentlys understands: extra metadata from the
        # server (annotations, icons, ...) is untrusted and unused.
        functions_schema = {
            "name": name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        }
        schemas.append(functions_schema)

        # add function that will encapsulate the call to the tool
        def call_tool_wrapper(function_name):
            async def call_tool(**kwargs):
                result = await mcp_server.call_tool(function_name, arguments=kwargs)
                return convert_tool_result(result, max_result_chars=max_result_chars)

            return call_tool

        # call_tool uses the server-side (unprefixed) tool name
        functions[name] = call_tool_wrapper(tool.name)

    return functions, schemas


async def fetch_mcp_server_resources(
    mcp_server: ClientSession,
    prefix: str = "",
    max_result_chars: typing.Optional[int] = None,
):
    """
    Fetch the resources from the MCP server and return (functions, schemas).
    """
    schemas = []
    functions = {}
    resources = (await mcp_server.list_resources()).resources
    resources += (await mcp_server.list_resource_templates()).resourceTemplates

    def extract_parameters_from_uri(uri: str) -> list:
        # example of uri: users://{user_id}/profile/{profile_id}
        # in this case, we want to return ["user_id", "profile_id"]
        return [
            part.strip("{").strip("}")
            for part in uri.split("/")
            if "{" in part and "}" in part
        ]

    for resource in resources:
        if getattr(resource, "uriTemplate", None):
            uri = str(resource.uriTemplate)
        elif getattr(resource, "uri", None):
            uri = str(resource.uri)
        else:
            raise ValueError(f"Invalid resource: {resource}")

        parameters = extract_parameters_from_uri(uri)
        description = "uri:" + uri
        name = sanitize_tool_name(
            uri.replace("://", "__")
            .replace("/", "_")
            .replace("{", "")
            .replace("}", ""),
            prefix,
        )
        if name in functions:
            logger.warning("Duplicate MCP resource name %r, skipping", name)
            continue
        if resource.description:
            description += "\n" + resource.description

        functions_schema = {
            "name": name,
            "description": description,
            "parameters": {
                "properties": {
                    param: {"title": param, "type": "string"} for param in parameters
                },
                "required": parameters,
                "type": "object",
            },
        }
        schemas.append(functions_schema)

        # add function that will encapsulate the call to the resource
        def read_resource_wrapper(uri_template):
            async def read_resource(**kwargs):
                # Reconstruct the uri from the function name
                uri = uri_template
                for key in kwargs:
                    uri = uri.replace("{" + key + "}", kwargs[key])
                result = await mcp_server.read_resource(uri)
                parts = []
                for content in getattr(result, "contents", None) or []:
                    text = getattr(content, "text", None)
                    if text is not None:
                        parts.append(text)
                    else:
                        parts.append("[binary content omitted]")
                text = "\n".join(parts) or "(empty result)"
                return _truncate_result(text, max_result_chars)

            return read_resource

        functions[name] = read_resource_wrapper(uri_template=uri)

    return functions, schemas
