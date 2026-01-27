"""Agentlys package."""

__version__ = "0.4.1"

import asyncio
import inspect
import io
import json
import logging
import os
import traceback
import typing
import warnings
from typing import Type, Union

from PIL import Image as PILImage

from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import APIProvider, BaseProvider
from agentlys.providers.utils import get_provider_and_model
from agentlys.utils import (
    csv_dumps,
    get_event_loop_or_create,
    inspect_schema,
    parse_chat_template,
)

AGENTLYS_HOST = os.getenv("AGENTLYS_HOST")
AGENTLYS_MODEL = os.getenv("AGENTLYS_MODEL")
OUTPUT_SIZE_LIMIT = int(os.getenv("AGENTLYS_OUTPUT_SIZE_LIMIT", 20_000))


def _truncate_with_warning(text: str, limit: int = OUTPUT_SIZE_LIMIT) -> str:
    """Truncate text to limit and add warning if truncated."""
    if len(text) > limit:
        logging.warning(f"Output truncated from {len(text)} to {limit} characters")
        return (
            text[:limit]
            + f"\n[Warning: Output truncated from {len(text)} to {limit} characters]"
        )
    return text


class StopLoopException(Exception):
    pass


def simple_response_default_callback(response: Message) -> Message:
    """
    This function is called when the response is a simple response (no function call).
    By default, the conversation will stop after a simple response.
    You can override this function to change this behavior.
    """
    raise StopLoopException("Stopping the conversation after a simple response")


class Agentlys(AgentlysBase):
    def __llm__(self):
        return self.name  # If an agent is used as a tool, it must have a name

    def __init__(
        self,
        name: str = None,
        instruction: str = None,
        examples: typing.Union[list[Message], None] = None,
        messages: typing.Union[list[Message], None] = None,
        context: str = None,
        max_interactions: int = 100,
        model=AGENTLYS_MODEL,
        provider: Union[str, Type[BaseProvider]] = APIProvider.OPENAI,
        use_tools_only: bool = False,
        mcp_servers: typing.Union[list[object], None] = [],
        thinking: typing.Optional[dict] = None,
    ) -> None:
        """
        Initialize the Agentlys instance.
        Args:
            use_tools_only: bool = False,
                If True, the chat will only use tools and not the LLM.
                This is a beta feature and may change in the future.
            thinking: Optional[dict] = None,
                Extended thinking configuration for Anthropic models.
                Example: {"type": "enabled", "budget_tokens": 10000}
        """
        self.provider, self.model = get_provider_and_model(
            self, provider, model
        )  # TODO: rename register ?
        self.simple_response_callback = simple_response_default_callback
        if use_tools_only:
            warnings.warn(
                "use_tools_only is a beta feature and may change in the future"
            )
        self.use_tools_only = use_tools_only
        self.client = None  # TODO:
        self.name = name
        self.instruction = instruction
        if examples is None:
            self.examples = []
        else:
            self.examples = examples

        if messages is None:
            self.messages = []
        else:
            self.messages = messages

        self.context = context
        self.max_interactions = max_interactions
        self.thinking = thinking
        self.functions_schema = []
        self.functions = {}
        self.tools = {}
        for m in mcp_servers:
            self.add_mcp_server(m)

    @classmethod
    def from_template(cls, chat_template: str, **kwargs):
        instruction, examples = parse_chat_template(chat_template)
        return cls(
            instruction=instruction,
            examples=examples,
            **kwargs,
        )

    def reset(self):
        """Reset the chat state.

        This will clear the chat history and reset the functions and tools.
        """
        self.messages = []
        self.functions_schema = []
        self.functions = {}
        self.tools = {}

    @property
    def last_message(self):
        if not self.messages:
            return None
        return self.messages[-1].content

    @property
    def last_tools_states(self) -> typing.Optional[str]:
        """We add the repr() of each tool to the system context"""
        # If there are no tools, return None
        if not self.tools:
            return None
        tool_reprs = []
        for tool_id, tool in self.tools.items():
            tool_name = f"{tool.__class__.__name__}-{tool_id}"
            if hasattr(tool, "__llm__"):
                tool_output = tool.__llm__()
            elif hasattr(tool, "__repr__"):
                tool_output = repr(tool)
            elif hasattr(tool, "__str__"):
                tool_output = str(tool)
            else:
                raise ValueError(
                    f"Tool {tool_name} has no __llm__, __repr__ or __str__ method"
                )
            tool_output = _truncate_with_warning(tool_output)
            tool_reprs.append(f"### {tool_name}\n{tool_output}")
        tool_context = "\n".join(tool_reprs)

        """
        ## Last Tools States
        ### Tool 1
        {state}
        ### Tool 2
        {state}
        --- End of Last Tools States ---
        """
        return f"## Last Tools States\n{tool_context}\n--- End of Last Tools States ---"

    def load_messages(self, messages: list[Message]):
        # Check order of messages (based on createdAt)
        # Oldest first (createdAt ASC)
        # messages = sorted(messages, key=lambda x: x.createdAt)
        self.messages = messages  # [message for message in messages]

    def add_function(
        self,
        function: typing.Callable,
        function_schema: typing.Optional[dict] = None,
    ):
        if function_schema is None:
            # We try to infer the function schema from the function
            function_schema = inspect_schema(function)

        self.functions_schema.append(function_schema)
        self.functions[function_schema["name"]] = function

    def add_tool(
        self, tool: typing.Union[type, object], tool_id: typing.Optional[str] = None
    ) -> str:
        """Add a tool class or instance to the chat instance and return the tool id"""
        if isinstance(tool, type):
            # If a class is provided, instantiate it
            tool = tool()

        if tool_id is None:
            tool_id = str(id(tool))

        self.tools[tool_id] = tool

        class_name = tool.__class__.__name__
        tool_name = f"{class_name}-{tool_id}"
        # Get all methods and functions from the tool
        functions = inspect.getmembers(tool, inspect.ismethod) + inspect.getmembers(
            tool, inspect.isfunction
        )
        for function_name, function in functions:
            if function_name.startswith("_"):  # Skip private methods
                continue
            function_schema = inspect_schema(function)
            function_schema["name"] = f"{tool_name}__{function_name}"
            self.add_function(function, function_schema)
        return tool_id

    def remove_tool(self, tool_id: str):
        del self.tools[tool_id]
        self.functions_schema = [
            schema for schema in self.functions_schema if schema["name"] != tool_id
        ]
        self.functions = {
            name: func for name, func in self.functions.items() if name != tool_id
        }

    async def add_mcp_server(self, mcp_server):
        """
        Add a MCP server to the agent instance.
        The MCP server will be used to call tools and resources.
        """
        from agentlys.mcp import fetch_mcp_server_resources, fetch_mcp_server_tools

        logging.warning("Experimental feature: MCP servers")
        tools_functions, schemas = await fetch_mcp_server_tools(mcp_server)
        self.functions.update(tools_functions)
        self.functions_schema.extend(schemas)
        resources_functions, schemas = await fetch_mcp_server_resources(mcp_server)
        self.functions.update(resources_functions)
        self.functions_schema.extend(schemas)

    async def ask_async(
        self,
        message: typing.Union[Message, str, None] = None,
        **kwargs,
    ) -> Message:
        """Async version of ask method that uses the async fetch_async provider method"""
        if message:
            if isinstance(message, str):
                # If message is instance of string, then convert to Message
                message = Message(
                    role="user",
                    content=message,
                )
            self.messages.append(message)  # Add the question to the history

        # Merge class-level thinking with any kwargs override
        if self.thinking and "thinking" not in kwargs:
            kwargs["thinking"] = self.thinking

        # Call the async strategy
        response = await self.provider.fetch_async(**kwargs)
        self.messages.append(response)
        return response

    def ask(
        self,
        message: typing.Union[Message, str, None] = None,
        **kwargs,
    ) -> Message:
        # For backward compatibility, use run_until_complete to execute the async method
        loop = get_event_loop_or_create()
        return loop.run_until_complete(self.ask_async(message, **kwargs))

    def _format_callback_message(
        self, function_name, function_call_id, content, image
    ):
        if isinstance(content, Message):
            message = content
            # TODO: Add name and function_call_id to the message
            # message.name = function_name
            # message.function_call_id = function_call_id
            return message

        # We format the function_call response to be used as a next message
        if content is None:
            # When content is None, use an empty string instead to prevent the "Message should have at least one part" error
            formatted_content = ""
        elif isinstance(content, list):
            formatted_content = []
            if not content:
                formatted_content = "[]"
            elif isinstance(content[0], dict):
                # If data is list of dicts, dumps to CSV
                formatted_content = csv_dumps(content, OUTPUT_SIZE_LIMIT)
            else:
                for item in content:
                    if isinstance(item, str):
                        formatted_content.append(item)
                    elif isinstance(item, object):
                        tool_id = self.add_tool(item)
                        formatted_content.append(f"Added tool: {tool_id}")
                    elif isinstance(item, (int, float, bool)):
                        formatted_content.append(str(item))
                    else:
                        raise ValueError(f"Invalid item type: {type(item)}")
                formatted_content = "\n".join(formatted_content)
                # Limit the size of the content
                if len(formatted_content) > OUTPUT_SIZE_LIMIT:
                    formatted_content = (
                        formatted_content[:OUTPUT_SIZE_LIMIT]
                        + f"\n... ({len(formatted_content)} characters)"
                    )
        elif isinstance(content, dict):
            content_dump = json.dumps(content)
            if len(content_dump) > OUTPUT_SIZE_LIMIT:
                formatted_content = (
                    content_dump[:OUTPUT_SIZE_LIMIT]
                    + f"\n... ({len(content_dump)} characters)"
                )
            else:
                formatted_content = content_dump
        elif isinstance(content, str):
            if len(content) > OUTPUT_SIZE_LIMIT:
                formatted_content = (
                    content[:OUTPUT_SIZE_LIMIT] + f"\n... ({len(content)} characters)"
                )
            else:
                formatted_content = content
        elif isinstance(content, bytes):
            # Detect if it's an image
            try:
                image = PILImage.open(io.BytesIO(content))
                formatted_content = None
            except IOError:
                # Not an image
                raise ValueError("Returned bytes is not a valid image.")
        elif isinstance(content, PILImage.Image):
            image = content
            formatted_content = None
        elif isinstance(content, (int, float, bool)):
            formatted_content = str(content)
        elif isinstance(content, object):
            # If the function return an object, we add it as a tool
            # NOTE: Maybe we shouldn't, and rely on a clearer signal / object type ?
            tool_id = self.add_tool(content)
            formatted_content = f"Added tool: {tool_id}"
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

        # Build a function response message.
        # If there's an image, create a single function_result_image part that includes
        # both the text content and the image. This ensures we have exactly ONE tool_result
        # per tool_use_id.
        if image:
            part = MessagePart(
                type="function_result_image",
                content=formatted_content,  # Include text content in the image part
                image=image,
                function_call_id=function_call_id,
            )
        else:
            part = MessagePart(
                type="function_result",
                content=formatted_content,
                function_call_id=function_call_id,
            )

        return Message(name=function_name, role="function", parts=[part])

    def _format_exception(self, e):
        # We clean the traceback to remove frames from __init__.py
        tb = traceback.extract_tb(e.__traceback__)
        filtered_tb = [frame for frame in tb if "chat.py" not in frame.filename]
        if filtered_tb:
            content = "Traceback (most recent call last):\n"
            content += "".join(traceback.format_list(filtered_tb))
            content += f"\n{e.__class__.__name__}: {str(e)}"
        else:
            # If no relevant frames, use the full traceback
            content = traceback.format_exc()
        return content

    async def _call_with_signature(self, func, from_response, **kwargs):
        sig = inspect.signature(func)
        if "from_response" in sig.parameters:
            result = func(**kwargs, from_response=from_response)
        else:
            result = func(**kwargs)

        # If the function is async, await the result
        if inspect.iscoroutine(result):
            return await result
        else:
            return result

    async def _resolve_and_call_function(
        self, name: str, args: dict, response: Message
    ) -> typing.Any:
        """Resolve function/tool by name and call it with args."""
        if name.startswith("tool-"):
            tool_id, method_name = name.split("__")
            tool = self.tools[tool_id]
            method = getattr(tool, method_name)
            return await self._call_with_signature(method, response, **args)
        else:
            return await self._call_with_signature(
                self.functions[name], response, **args
            )

    def _process_tool_result(
        self, result: typing.Any
    ) -> tuple[typing.Any, typing.Any]:
        """Process tool result and handle tuple unpacking.

        Returns:
            Tuple of (content, image)
        """
        content = None
        image = None

        # Handle functions that return (content, image) tuples
        if isinstance(result, tuple) and len(result) == 2:
            content, image = result
        else:
            content = result

        return content, image

    async def _call_function_and_build_message(
        self, function_name, function_arguments, response
    ):
        """
        Encapsulate the 'call the function' logic & handle exceptions
        plus building a function or tool response message.
        """
        content = None
        image = None

        try:
            result = await self._resolve_and_call_function(
                function_name, function_arguments, response
            )
            content, image = self._process_tool_result(result)
        except StopLoopException:
            raise
        except Exception as e:
            content = self._format_exception(e)

        # Build next message
        return self._format_callback_message(
            function_name=function_name,
            function_call_id=response.function_call_id,
            content=content,
            image=image,
        )

    async def _execute_single_tool(
        self,
        part: MessagePart,
        response: Message,
    ) -> tuple[str, str, typing.Any, typing.Any]:
        """Execute one tool and return (function_call_id, function_name, content, image)."""
        name = part.function_call["name"]
        args = part.function_call["arguments"]
        function_call_id = part.function_call_id

        content = None
        image = None

        try:
            result = await self._resolve_and_call_function(name, args, response)
            content, image = self._process_tool_result(result)
        except StopLoopException:
            raise  # Propagate to stop the loop
        except Exception as e:
            content = self._format_exception(e)

        return (function_call_id, name, content, image)

    async def _call_functions_parallel(
        self,
        function_call_parts: list[MessagePart],
        response: Message,
    ) -> Message:
        """
        Execute multiple tool calls in parallel and build a combined result message.

        Per Anthropic API requirements, all tool_results must be in a SINGLE user message.
        Uses asyncio.gather with return_exceptions=True to collect all results even if some fail.
        """
        # Execute all tools in parallel
        results = await asyncio.gather(
            *[
                self._execute_single_tool(part, response)
                for part in function_call_parts
            ],
            return_exceptions=True,  # Don't fail fast - collect all results
        )

        # Build combined message with all tool results
        parts = []
        formatted_messages = []
        for i, result in enumerate(results):
            if isinstance(result, StopLoopException):
                raise result
            # _execute_single_tool handles its own exceptions,
            # so this path is rare (e.g., task-level failures)
            if isinstance(result, Exception):
                function_call_id = function_call_parts[i].function_call_id
                function_name = function_call_parts[i].function_call["name"]
                error_content = self._format_exception(result)
                parts.append(
                    MessagePart(
                        type="function_result",
                        content=error_content,
                        function_call_id=function_call_id,
                    )
                )
                continue

            function_call_id, function_name, content, image = result
            formatted_msg = self._format_callback_message(
                function_name=function_name,
                function_call_id=function_call_id,
                content=content,
                image=image,
            )
            formatted_messages.append(formatted_msg)
            # Extract parts from formatted message
            parts.extend(formatted_msg.parts)

        # For single function call, return the formatted message directly
        # to preserve the name attribute (important for OpenAI compatibility)
        if len(formatted_messages) == 1 and len(parts) == len(
            formatted_messages[0].parts
        ):
            return formatted_messages[0]

        # Create single message with all results for multiple tools
        return Message(role="function", parts=parts)

    async def _stream_functions_parallel(
        self,
        function_call_parts: list[MessagePart],
        response: Message,
    ) -> typing.AsyncGenerator[tuple[str, str, Message], None]:
        """
        Execute multiple tool calls in parallel, yielding results as each completes.

        Yields tuples of (function_call_id, function_name, result_message) as each tool finishes.
        This allows streaming individual results to the UI while tools execute in parallel.
        """
        # Create tasks - store as a list for asyncio.wait
        tasks = [
            asyncio.create_task(self._execute_single_tool(part, response))
            for part in function_call_parts
        ]

        # Use asyncio.wait with FIRST_COMPLETED to yield results as they finish
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    result = task.result()
                except StopLoopException:
                    # Cancel remaining tasks and await them before raising
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    raise
                except Exception as e:
                    # Handle unexpected exceptions - _execute_single_tool handles
                    # its own exceptions, so this path is rare
                    error_msg = Message(
                        role="function",
                        content=self._format_exception(e),
                        function_call_id=None,
                    )
                    yield (None, "unknown", error_msg)
                    continue

                function_call_id, function_name, content, image = result
                formatted_msg = self._format_callback_message(
                    function_name=function_name,
                    function_call_id=function_call_id,
                    content=content,
                    image=image,
                )
                yield (function_call_id, function_name, formatted_msg)

    async def run_conversation_async(
        self,
        question: typing.Union[str, Message, None] = None,
    ) -> typing.AsyncGenerator[Message, None]:
        """Async version of run_conversation with parallel tool execution support."""
        # If there's an initial question, emit it:
        if isinstance(question, str):
            message = Message(
                role="user",
                content=question,
            )
            yield message
        else:
            message = question

        for _ in range(self.max_interactions):
            # Ask the LLM with the current message (if any)
            response = await self.ask_async(message)

            # Get ALL function_call parts (supports parallel tool calls)
            function_call_parts = response.function_call_parts

            if not function_call_parts:
                yield response
                try:
                    message = self.simple_response_callback(response)
                except StopLoopException:
                    return
            else:
                yield response

                try:
                    # Execute all tools in parallel
                    message = await self._call_functions_parallel(
                        function_call_parts, response
                    )
                except StopLoopException:
                    return

                yield message

    async def ask_stream_async(
        self,
        message: typing.Union[Message, str, None] = None,
        **kwargs,
    ) -> typing.AsyncGenerator[dict, None]:
        """Async streaming version of ask method.

        Yields:
            - {"type": "text", "content": str} - text chunks as they arrive
            - {"type": "message", "message": Message} - final complete message
        """
        if message:
            if isinstance(message, str):
                message = Message(role="user", content=message)
            self.messages.append(message)

        final_message = None
        async for chunk in self.provider.fetch_stream_async(**kwargs):
            if chunk["type"] == "message":
                final_message = chunk["message"]
                self.messages.append(final_message)
            yield chunk

        if final_message is None:
            raise RuntimeError("Stream ended without final message")

    async def run_conversation_stream_async(
        self,
        question: typing.Union[str, Message, None] = None,
    ) -> typing.AsyncGenerator[dict, None]:
        """Run conversation with streaming text responses and parallel tool execution.

        Yields:
            - {"type": "user", "message": Message} - user message
            - {"type": "text", "content": str} - text chunks as they stream
            - {"type": "assistant", "message": Message} - complete assistant message
            - {"type": "tools_executing", "data": dict} - signal parallel tool execution start
            - {"type": "tool_result", "data": dict} - individual tool result as it completes
            - {"type": "function", "message": Message} - combined function results message
        """
        # Handle initial question
        if isinstance(question, str):
            message = Message(role="user", content=question)
            yield {"type": "user", "message": message}
        else:
            message = question

        for _ in range(self.max_interactions):
            # Stream the LLM response
            response = None
            async for chunk in self.ask_stream_async(message):
                if chunk["type"] == "message":
                    response = chunk["message"]
                else:
                    # Forward all other events (text, tool_started, tool_delta, etc.)
                    yield chunk

            if response is None:
                raise RuntimeError("Stream ended without response")

            # Get ALL function_call parts (supports parallel tool calls)
            function_call_parts = response.function_call_parts

            if not function_call_parts:
                yield {"type": "assistant", "message": response}
                try:
                    message = self.simple_response_callback(response)
                except StopLoopException:
                    return
            else:
                # Yield assistant message with all tool calls
                yield {"type": "assistant", "message": response}

                # Signal start of parallel execution
                yield {
                    "type": "tools_executing",
                    "data": {
                        "count": len(function_call_parts),
                        "tools": [
                            {
                                "name": p.function_call["name"],
                                "id": p.function_call_id,
                            }
                            for p in function_call_parts
                        ],
                    },
                }

                try:
                    # Execute tools in parallel, streaming results as they complete
                    all_parts = []
                    result_messages = []
                    async for (
                        function_call_id,
                        function_name,
                        result_msg,
                    ) in self._stream_functions_parallel(function_call_parts, response):
                        # Yield individual result for UI streaming
                        yield {
                            "type": "tool_result",
                            "data": {
                                "function_call_id": function_call_id,
                                "function_name": function_name,
                                "message": result_msg,
                            },
                        }
                        # Collect for combined message
                        result_messages.append(result_msg)
                        all_parts.extend(result_msg.parts)

                    # Build combined message for conversation history
                    # For single function call, use the formatted message directly
                    # to preserve the name attribute (important for OpenAI legacy)
                    if len(result_messages) == 1:
                        message = result_messages[0]
                    else:
                        message = Message(role="function", parts=all_parts)

                except StopLoopException:
                    return

                # Yield combined function results (for consumers that expect it)
                yield {"type": "function", "message": message}

    def run_conversation(self, question: typing.Union[str, Message, None] = None):
        # For backward compatibility, use run_until_complete to execute the async method
        loop = get_event_loop_or_create()
        async_gen = self.run_conversation_async(question)

        while True:
            try:
                # Get the next item from the async generator
                item = loop.run_until_complete(async_gen.__anext__())
                # Yield the item immediately
                yield item
            except StopAsyncIteration:
                # Stop when the async generator is exhausted
                break
