import json
import os
import typing

from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider
from agentlys.providers.utils import FunctionCallParsingError

AGENTLYS_HOST = os.getenv("AGENTLYS_HOST")


def from_openai_object(
    role: str,
    content: str,
    function_call: typing.Optional[dict] = None,
    id: typing.Optional[str] = None,
):
    # We need to unquote the arguments
    function_call_dict = None
    if function_call:
        try:
            function_call_dict = {
                "name": function_call.name,
                "arguments": json.loads(function_call.arguments),
            }
        except json.decoder.JSONDecodeError:
            raise FunctionCallParsingError(id, function_call)

    return Message(role=role, content=content, function_call=function_call_dict, id=id)


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
    res = {
        "role": message.role,
        "content": [],
    }

    for part in message.parts:
        if part.type == "function_call" and message.role == "assistant":
            res["function_call"] = parts_to_openai_dict(part)
        elif part.type == "function_call" and message.role == "user":
            # Workaround: If the user is triggering a function, we add it's name and arguments to the content
            res["content"] = (
                part.function_call["name"]
                + ":"
                + json.dumps(part.function_call["arguments"])
            )
        else:
            res["content"].append(parts_to_openai_dict(part))

    if message.name:
        res["name"] = message.name

    # Workaround
    # Image URLs are only allowed for messages with role 'user'
    # If the message contains an image, we change it's role to 'user'
    if message.role == "function" and any(
        part.type == "function_result_image" for part in message.parts
    ):
        res["role"] = "user"
    return res


class OpenAIProviderFunctionLegacy(BaseProvider):
    def __init__(self, chat: AgentlysBase, model: str, base_url: str = None):
        from openai import OpenAI

        self.chat = chat
        self.model = model
        # Possibly set up openai.api_key, base_url, etc.
        self.client = OpenAI(
            base_url=(
                f"{AGENTLYS_HOST}/v1" if AGENTLYS_HOST else "https://api.openai.com/v1"
            ),
        )

    async def fetch_async(self, **kwargs) -> Message:
        messages = self.prepare_messages(transform_function=message_to_openai_dict)
        # Add instruction as the first message

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

        if self.chat.functions_schema:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.chat.functions_schema,
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
            function_call=message.function_call,
            id=res.id,  # We use the response id as the message id
        )
