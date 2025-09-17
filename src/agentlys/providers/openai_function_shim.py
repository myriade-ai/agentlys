"""
Temporary provider so we can use function calls with OpenAI models
that don't support them yet in the API (ie. o1).
"""

import json
import typing

from agentlys.model import Message
from agentlys.providers.openai import OpenAIProvider, parts_to_openai_dict


def parse_function_call_from_text(text: str):
    import re

    """
    Look for something like:

    CALLING FUNCTION: my_function
    {
        "foo": "bar"
    }

    Return (function_name, arguments_dict) or (None, None) if no function call found.
    This is a simplistic regex approach. Adjust as needed.
    """
    pattern = r"(?s)CALLING\s+FUNCTION:\s*([A-Za-z0-9_]+)\s*\n?\s*(\{.*?\})"
    match = re.search(pattern, text)
    if match:
        function_name = match.group(1).strip()
        try:
            # Attempt to parse JSON in the curly braces
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            arguments = {}
        return function_name, arguments
    return None, None


def from_openai_object(
    role: str,
    content: str,
    id: typing.Optional[str] = None,
):
    function_name, function_args = parse_function_call_from_text(content)
    function_call = None
    if function_name:
        function_call = {
            "name": function_name,
            "arguments": function_args,
        }
        # Shortcut
        content = None
    return Message(role=role, content=content, function_call=function_call, id=id)


def message_to_openai_dict(message: Message) -> dict:
    """o1 only supports text-based instructions for function usage"""
    res = {
        "role": "user" if message.role == "function" else message.role,
        "content": [],
    }

    for part in message.parts:
        if part.type == "function_call":
            text = f"CALLING FUNCTION: {part.function_call['name']}\n"
            text += f"{json.dumps(part.function_call['arguments'])}\n"
            res["content"].append({"type": "text", "text": text})
        else:
            res["content"].append(parts_to_openai_dict(part))

    return res


class OpenAIProviderFunctionShim(OpenAIProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def fetch_async(self):
        """
        Calls the OpenAI ChatCompletion endpoint using ONLY text-based instructions for function usage.
        """

        messages = self.prepare_messages(transform_function=message_to_openai_dict)
        # Add instruction as the first message
        if self.chat.instruction:
            instruction_message = Message(
                role="user",
                content=self.chat.instruction,
            )
            messages = [message_to_openai_dict(instruction_message)] + messages

        # Inject function schema into the system message, if we haven't already
        # or just append it to an existing system message. Example:
        if self.chat.functions_schema:
            function_schemas_text = json.dumps(self.chat.functions_schema, indent=2)
            augmented_instruction = (
                (self.chat.instruction or "")
                + "\n\nHere are the available functions you can call, in JSON format:\n"
                + function_schemas_text
                + "\n\n"
                + "When you call a function, respond with:\n"
                + "```\n"
                + "CALLING FUNCTION: <function_name>\n"
                + "{\n"
                + '  "arg1": "...",\n'
                + '  "arg2": "..."\n'
                + "}\n"
                + "```\n"
                + "If you do not need to call a function, just respond normally."
            )
        else:
            augmented_instruction = self.chat.instruction

        if augmented_instruction:
            system_msg = Message(
                role="user",
                content=augmented_instruction,
            )
            messages = [message_to_openai_dict(system_msg)] + messages

        # Make the request WITHOUT 'functions='
        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        # OpenAI returns a structure. We'll parse out the text content
        message = res.choices[0].message
        # Note that now there is no function_call field because we didn't pass `functions=`.
        return from_openai_object(
            role=message.role,
            content=message.content,
            # no function_call here
            id=res.id,
        )
