import typing
from abc import ABC, abstractmethod
from enum import Enum

from agentlys.model import Message, MessagePart
from agentlys.utils import get_event_loop_or_create


class APIProvider(Enum):
    OPENAI = "openai"
    OPENAI_FUNCTION_LEGACY = "openai_function_legacy"
    OPENAI_FUNCTION_SHIM = "openai_function_shim"
    ANTHROPIC = "anthropic"
    DEFAULT = "default"


class BaseProvider(ABC):
    def prepare_messages(
        self,
        transform_function: typing.Callable,
        transform_list_function: typing.Callable = lambda x: x,
    ) -> list[dict]:
        """Prepare messages for API requests using a transformation function."""
        first_message = self.chat.messages[0]

        # Add combined context to the first message if it exists
        if self.chat.context:
            if isinstance(first_message.content, str):
                first_message.parts[0].content = (
                    self.chat.context + "\n" + first_message.parts[0].content
                )
            elif isinstance(first_message.content, list):
                first_message.content = [
                    MessagePart(type="text", content=self.chat.context),
                    *first_message.content,
                ]

        messages = self.chat.examples + [first_message] + self.chat.messages[1:]
        messages = transform_list_function(messages)
        return [transform_function(m) for m in messages]

    @abstractmethod
    async def fetch_async(self, **kwargs) -> Message:
        """
        Async version of fetch method.
        Given a chat context, returns a single new Message from the LLM.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def fetch(self, **kwargs) -> Message:
        """
        Given a chat context, returns a single new Message from the LLM.
        """
        # For backward compatibility, use run_until_complete to execute the async method
        loop = get_event_loop_or_create()
        return loop.run_until_complete(self.fetch_async(**kwargs))
