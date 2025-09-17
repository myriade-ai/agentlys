import json
import os
import typing

from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider
from agentlys.providers.utils import FunctionCallParsingError, add_empty_function_result

AGENTLYS_HOST = os.getenv("AGENTLYS_HOST")


def from_openai_object(
    role: str,
    content: str,
    tool_calls: typing.Optional[list] = None,
    id: typing.Optional[str] = None,
):
    # We need to unquote the arguments

    function_call_dict = None
    function_call_id = None
    if tool_calls:
        first_tool_call_function = next(
            (t for t in tool_calls if t.type == "function"), None
        )
        if first_tool_call_function is None:
            raise ValueError(
                "We don't support tool calls with type other than function"
            )
        function_call = first_tool_call_function.function
        function_call_id = first_tool_call_function.id
        try:
            function_call_dict = {
                "name": function_call.name,
                "arguments": json.loads(function_call.arguments),
            }
        except json.decoder.JSONDecodeError:
            raise FunctionCallParsingError(id, function_call)

    return Message(
        role=role,
        content=content,
        function_call=function_call_dict,
        id=id,
        function_call_id=function_call_id,
    )


def parts_to_openai_dict(part: MessagePart) -> dict:
    if part.type == "text":
        return {
            "type": "text",
            "text": part.content,
        }
    elif part.type == "image":
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{part.image.format};base64,{part.image.to_base64()}"
            },
        }
    elif part.type == "function_call":
        return {
            "name": part.function_call["name"],
            "arguments": json.dumps(part.function_call["arguments"]),
        }
    elif part.type == "function_result":
        return {
            "type": "text",
            "text": part.content,
        }
    elif part.type == "function_result_image":
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{part.image.format};base64,{part.image.to_base64()}"
            },
        }

    raise ValueError(f"Unknown part type: {part.type}")


def message_to_openai_dict(message: Message) -> dict:
    if message.role == "function":
        res = {
            "role": "tool",
            "tool_call_id": message.function_call_id,
            "content": [parts_to_openai_dict(part) for part in message.parts],
            "name": message.name,
        }
    else:
        res = {"role": message.role, "content": []}
        for part in message.parts:
            if part.type == "function_call" and message.role == "assistant":
                res["tool_calls"] = [
                    {
                        "id": part.function_call_id,
                        "type": "function",
                        "function": parts_to_openai_dict(part),
                    }
                ]
            elif part.type == "function_call" and message.role == "user":
                # Workaround: If the user is triggering a function, we add it's name and arguments to the content
                res["content"] = (
                    part.function_call["name"]
                    + ":"
                    + json.dumps(part.function_call["arguments"])
                )
            else:
                res["content"].append(parts_to_openai_dict(part))

        if "content" in res and len(res["content"]) == 0:
            res["content"] = None

    return res


def return_image_as_user_message(messages: list[Message]) -> list[Message]:
    """
    Adaptation because OpenAI doesn't support image in function call.
    We return the image as a user message.
    """
    for i in range(len(messages)):
        if messages[i].role == "function" and messages[i].image is not None:
            messages[i].role = "user"
    return messages


class OpenAIProvider(BaseProvider):
    def __init__(self, chat: AgentlysBase, model: str, base_url: str = None):
        from openai import OpenAI

        self.chat = chat
        self.model = model
        # Possibly set up openai.api_key, base_url, etc.
        self.client = OpenAI(
            base_url=(AGENTLYS_HOST if AGENTLYS_HOST else "https://api.openai.com/v1"),
        )

    async def fetch_async(self, **kwargs) -> Message:
        messages = self.prepare_messages(
            transform_function=message_to_openai_dict,
            transform_list_function=lambda x: add_empty_function_result(
                return_image_as_user_message(x)
            ),
        )

        system_messages = []
        if self.chat.instruction:
            system_messages.append(
                Message(
                    role="system",
                    content=self.chat.instruction,
                )
            )
        if self.chat.last_tools_states:
            system_messages.append(
                Message(
                    role="system",
                    content=self.chat.last_tools_states,
                )
            )
        messages = [message_to_openai_dict(sm) for sm in system_messages] + messages

        if self.chat.use_tools_only and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = "required"

        if self.chat.functions_schema:
            tools = [
                {
                    "type": "function",
                    "function": tool,
                }
                for tool in self.chat.functions_schema
            ]
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        else:
            res = self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )

        message = res.choices[0].message
        return from_openai_object(
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            id=res.id,  # We use the response id as the message id
        )
