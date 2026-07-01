import os
from typing import Type, Union

from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import APIProvider, BaseProvider


class FunctionCallParsingError(Exception):
    def __init__(self, id, function_call):
        self.id = id
        self.function_call = function_call

    def __str__(self):
        return f"Invalid function_call: {self.obj.function_call}"


# TODO: should probably exploit default model from provider
def get_provider_and_model(  # TODO: get_provider_and_model ?
    # chat: Agentlys, # TODO: make AgentlysBase ?
    chat,
    provider_name: Union[str, Type[BaseProvider]] = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> list[str, BaseProvider]:  # TODO: rename
    """
    Returns the correct LLM provider based on a string or env vars.

    ``base_url`` and ``api_key`` let you target any OpenAI-compatible API
    (Ollama, vLLM, LiteLLM, OpenRouter, Azure OpenAI, ...) or a custom
    Anthropic-compatible endpoint.
    """

    if not provider_name:
        provider_name = os.getenv("AGENTLYS_PROVIDER", "openai")

    # Only forward what was explicitly provided so custom provider classes
    # that don't accept base_url/api_key keep working.
    client_kwargs = {}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key is not None:
        client_kwargs["api_key"] = api_key

    if isinstance(provider_name, type) and issubclass(provider_name, BaseProvider):
        # Supports custom provider
        Provider = provider_name
        return Provider(chat, model=model, **client_kwargs), model
    elif isinstance(provider_name, APIProvider):
        provider_key = provider_name
    elif isinstance(provider_name, str):
        try:
            provider_key = APIProvider(provider_name)
        except ValueError:
            raise ValueError(f"Provider {provider_name} is not a valid provider")
    else:
        raise ValueError(f"Invalid provider: {provider_name}")

    if provider_key == APIProvider.OPENAI:
        from agentlys.providers.openai import OpenAIProvider

        if not model:
            model = os.getenv("AGENTLYS_MODEL", "gpt-4o")
        return OpenAIProvider(chat, model=model, **client_kwargs), model
    elif provider_key == APIProvider.ANTHROPIC:
        from agentlys.providers.anthropic import AnthropicProvider

        if not model:
            model = os.getenv("AGENTLYS_MODEL", "claude-3-7-sonnet-latest")
        return AnthropicProvider(chat, model=model, **client_kwargs), model
    elif provider_key == APIProvider.OPENAI_FUNCTION_SHIM:
        from agentlys.providers.openai_function_shim import OpenAIProviderFunctionShim

        if not model:
            model = os.getenv("AGENTLYS_MODEL", "o1-preview")
        return OpenAIProviderFunctionShim(chat, model=model, **client_kwargs), model
    elif provider_key == APIProvider.OPENAI_FUNCTION_LEGACY:
        from agentlys.providers.openai_function_legacy import (
            OpenAIProviderFunctionLegacy,
        )

        if not model:
            model = os.getenv("AGENTLYS_MODEL", "gpt-4o")
        return OpenAIProviderFunctionLegacy(chat, model=model, **client_kwargs), model
    elif provider_key == APIProvider.DEFAULT:
        from agentlys.providers.default import DefaultProvider

        if not model:
            raise ValueError("Default provider requires a model")
        return DefaultProvider(chat, model=model, **client_kwargs), model
    else:
        raise ValueError(f"Provider {provider_key} is not supported")


def add_empty_function_result(messages: list[Message]) -> list[Message]:
    """
    OpenAI/Anthropic requires a call to have a result message. Not what we want.
    Adjustment for merging or inserting the "function_result":

    - First case: a message with `role="function"` followed by a message with `role="user"`.
      We transform this 'function' message into a part of type 'tool_result' and insert it at the beginning of the following user message.

    - Second case (unchanged): a message with `role="assistant"` containing a `function_call`, followed by a non-`function` message.
      We insert an empty message with `role="function"`.
      Supports parallel tool calls by inserting empty results for all function calls.
    """
    for i in range(len(messages) - 1, 0, -1):
        function_call_parts = messages[i - 1].function_call_parts
        if (
            messages[i - 1].role == "assistant"
            and function_call_parts
            and not messages[i].role == "function"
        ):
            # Insert empty function results for all function calls
            empty_result_parts = [
                MessagePart(
                    type="function_result",
                    content="",
                    function_call_id=part.function_call_id,
                )
                for part in function_call_parts
            ]
            messages.insert(
                i,
                Message(
                    role="function",
                    parts=empty_result_parts,
                ),
            )
    return messages
