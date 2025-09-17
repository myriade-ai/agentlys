import json
import os

from agentlys.base import AgentlysBase
from agentlys.model import Message
from agentlys.providers.openai import (
    OpenAIProvider,
    parts_to_openai_dict,
)

AGENTLYS_HOST = os.getenv("AGENTLYS_HOST")


def message_to_openai_dict(message: Message) -> dict:
    if message.role == "function":
        res = {
            "role": "tool",
            "tool_call_id": message.function_call_id,
            "content": message.content,
            "name": message.name,
        }
    elif message.role == "system":
        res = {"role": message.role, "content": message.content}
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


class DefaultProvider(OpenAIProvider):
    """Default provider is OpenAI with small changes
    - System messages are string only
    - Function call content is a string
    """

    def __init__(self, chat: AgentlysBase, model: str, base_url: str = None):
        from openai import OpenAI

        self.chat = chat
        self.model = model
        self.client = OpenAI(base_url=AGENTLYS_HOST)
