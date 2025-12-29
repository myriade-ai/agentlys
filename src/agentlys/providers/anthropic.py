import os

import anthropic
from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider
from agentlys.providers.utils import add_empty_function_result

AGENTLYS_HOST = os.getenv("AGENTLYS_HOST")


def part_to_anthropic_dict(part: MessagePart) -> dict:
    if part.type == "text":
        return {
            "type": "text",
            "text": part.content,
        }
    elif part.type == "image":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.image.format,
                "data": part.image.to_base64(),
            },
        }
    elif part.type == "function_call":
        return {
            "type": "tool_use",
            "id": part.function_call_id,
            "name": part.function_call["name"],
            "input": part.function_call["arguments"],
        }
    elif part.type == "function_result_image":
        return {
            "type": "tool_result",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.image.format,
                        "data": part.image.to_base64(),
                    },
                }
            ],
            "tool_use_id": part.function_call_id,
        }
    elif part.type == "function_result":
        return {
            "type": "tool_result",
            "tool_use_id": part.function_call_id,
            "content": part.content,
        }
    raise ValueError(f"Unknown part type: {part.type}")


def message_to_anthropic_dict(message: Message) -> dict:
    res = {
        "role": message.role if message.role in ["user", "assistant"] else "user",
        "content": [],
    }

    for part in message.parts:
        res["content"].append(part_to_anthropic_dict(part))

    return res


DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "10000"))


class AnthropicProvider(BaseProvider):
    def __init__(
        self, chat: AgentlysBase, model: str, max_tokens: int | None = None
    ):
        self.model = model
        self.client = anthropic.AsyncAnthropic(
            base_url=AGENTLYS_HOST if AGENTLYS_HOST else "https://api.anthropic.com",
        )
        self.chat = chat
        self.max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    async def fetch_async(self, **kwargs):
        messages = self.prepare_messages(
            transform_function=lambda m: message_to_anthropic_dict(m),
            transform_list_function=add_empty_function_result,
        )

        if self.chat.instruction:
            system = self.chat.instruction
        else:
            system = None

        def merge_messages(messages):
            """
            When two messages are in the same role, we merge the following message into the previous.
            {
                "role": "user",
                "content": [
                    {
                    "type": "tool_result",
                    "tool_use_id": "example_19",
                    "content": ""
                    }
                ]
            },
            {
                "role": "user",
                "content": "Plot distribution of stations per city"
            }
            """
            merged_messages = []
            for message in messages:
                if merged_messages and merged_messages[-1]["role"] == message["role"]:
                    if isinstance(merged_messages[-1]["content"], str):
                        merged_messages[-1]["content"].append(
                            {
                                "type": "text",
                                "text": merged_messages[-1]["content"],
                            }
                        )
                    elif isinstance(merged_messages[-1]["content"], list):
                        merged_messages[-1]["content"].extend(message["content"])
                    else:
                        raise ValueError(
                            f"Invalid content type: {type(merged_messages[-1]['content'])}"
                        )
                else:
                    merged_messages.append(message)
            return merged_messages

        messages = merge_messages(messages)

        # Need to map field "parameters" to "input_schema"
        tools = [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["parameters"],
            }
            for s in self.chat.functions_schema
        ]
        # Add description to the function is their description is empty
        for tool in tools:
            if not tool["description"]:
                tool["description"] = "No description provided"

        # === Add cache_controls ===
        # Messages: Find the last message with an index multiple of 10
        last_message_index = next(
            (i for i in reversed(range(len(messages))) if i % 10 == 0),
            None,
        )

        if last_message_index is not None:
            # Only try to modify the cache control if there are messages and content
            if (
                messages
                and isinstance(messages[last_message_index]["content"], list)
                and len(messages[last_message_index]["content"]) > 0
                and isinstance(messages[last_message_index]["content"][-1], dict)
            ):
                messages[last_message_index]["content"][-1]["cache_control"] = {
                    "type": "ephemeral"
                }
            elif isinstance(messages[last_message_index]["content"], str):
                messages[last_message_index]["content"] = [
                    {
                        "type": "text",
                        "text": messages[last_message_index]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
        # Tools: Add cache_control to the last tool function
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        # System: add cache_control to the system message
        system_messages = []
        if system is not None:
            system_messages.append(
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        if self.chat.last_tools_states:
            # add to system message
            system_messages.append(
                {
                    "type": "text",
                    "text": self.chat.last_tools_states,
                }
            )

        # === End of cache_control ===

        if system_messages:
            kwargs["system"] = system_messages

        if self.chat.use_tools_only and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = {"type": "any"}

        res = await self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        res_dict = res.to_dict()
        return Message.from_anthropic_dict(
            role=res_dict["role"],
            content=res_dict["content"],
        )
