from mcp import ClientSession


async def fetch_mcp_server_tools(mcp_server: ClientSession):
    """
    Fetch the tools from the MCP server and add them to the chat instance.
    """
    schemas = []
    functions = {}
    tools = (await mcp_server.list_tools()).tools

    for tool in tools:
        functions_schema = tool.model_dump()
        functions_schema["parameters"] = functions_schema.pop("inputSchema")
        schemas.append(functions_schema)

        # add function that will encapsulate the call to the tool
        def call_tool_wrapper(function_name):
            async def call_tool(**kwargs):
                result = await mcp_server.call_tool(function_name, arguments=kwargs)
                # NOTE: for now, we only return the text of the result
                return result.content[0].text

            return call_tool

        call_tool = call_tool_wrapper(functions_schema["name"])
        functions[functions_schema["name"]] = call_tool

    return functions, schemas


async def fetch_mcp_server_resources(mcp_server: ClientSession):
    """
    Fetch the resources from the MCP server and add them to the chat instance.
    """
    schemas = []
    functions = {}
    resources = (await mcp_server.list_resources()).resources
    resources += (await mcp_server.list_resource_templates()).resourceTemplates

    def extract_parameters_from_uri(uri: str) -> dict:
        # example of uri: users://{user_id}/profile/{profile_id}
        # in this case, we want to return ["user_id", "profile_id"]
        return [
            part.strip("{").strip("}")
            for part in uri.split("/")
            if "{" in part and "}" in part
        ]

    for resource in resources:
        if hasattr(resource, "uriTemplate"):
            uri = str(resource.uriTemplate)
        elif hasattr(resource, "uri"):
            uri = str(resource.uri)
        else:
            raise ValueError(f"Invalid resource: {resource}")

        parameters = extract_parameters_from_uri(uri)
        description = "uri:" + uri
        name = (
            uri.replace("://", "__").replace("/", "_").replace("{", "").replace("}", "")
        )
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

        # add function that will encapsulate the call to the tool
        def read_resource_wrapper(uri_template):
            async def read_resource(**kwargs):
                # Reconstruct the uri from the function name
                uri = uri_template
                for key in kwargs:
                    uri = uri.replace("{" + key + "}", kwargs[key])
                result = await mcp_server.read_resource(uri)
                # NOTE: for now, we only return the text of the result
                return result.contents[0].text

            return read_resource

        functions[name] = read_resource_wrapper(uri_template=uri)

    return functions, schemas
