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
        """Prepare messages for API requests using a transformation function.

        Context and instruction are not included here — each provider adds
        them to the system prompt natively (e.g. Anthropic's ``system``
        field, OpenAI's system messages).

        ``user_context`` (untrusted, user-provided content) is prepended to
        the last user message so the model sees it as user input, not as
        system instructions.
        """
        all_messages = self.chat.examples + self.chat.messages

        # Prepend user_context to the last user message.  Build a new
        # Message to avoid mutating the original (prepare_messages is
        # called on every LLM round-trip within a tool loop).
        if self.chat.user_context and all_messages:
            last_user_idx = None
            for i in range(len(all_messages) - 1, -1, -1):
                if all_messages[i].role == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                orig = all_messages[last_user_idx]
                context_part = MessagePart(
                    type="text", content=self.chat.user_context
                )
                patched = Message(
                    role=orig.role,
                    name=orig.name,
                    id=orig.id,
                    parts=[context_part, *orig.parts],
                )
                all_messages = (
                    all_messages[:last_user_idx]
                    + [patched]
                    + all_messages[last_user_idx + 1 :]
                )

        messages = all_messages
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

    async def fetch_stream_async(self, **kwargs) -> typing.AsyncGenerator[dict, None]:
        """
        Async streaming version of fetch method.
        Yields chunks as they arrive from the LLM.

        Yields:
            - {"type": "text", "content": str} - text chunks as they arrive
            - {"type": "message", "message": Message} - final complete message

        Note: Subclasses should override this method to provide streaming support.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming. "
            "Please implement fetch_stream_async or use a provider that supports streaming."
        )
        # This yield is needed to make this a generator function
        yield  # type: ignore  # pragma: no cover
