import typing
from abc import ABC, abstractmethod
from enum import Enum

from agentlys.model import Message, MessagePart
from agentlys.utils import get_event_loop_or_create


class APIProvider(Enum):
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_FUNCTION_LEGACY = "openai_function_legacy"
    OPENAI_FUNCTION_SHIM = "openai_function_shim"
    ANTHROPIC = "anthropic"
    DEFAULT = "default"


class EmptyCompletionError(RuntimeError):
    """A one-shot ``complete()`` call returned no text to use.

    Typically the model spent all of ``max_tokens`` thinking
    (``stop_reason="max_tokens"`` with only a thinking block), or declined
    (``stop_reason="refusal"``). ``stop_reason`` and ``block_types`` describe
    what came back instead, for logging.
    """

    def __init__(self, stop_reason: typing.Optional[str], block_types: list[str]):
        self.stop_reason = stop_reason
        self.block_types = block_types
        super().__init__(
            f"Completion response contained no text "
            f"(stop_reason={stop_reason}, blocks={block_types})"
        )


def completion_text(message: Message) -> str:
    """The text of a one-shot reply, or ``EmptyCompletionError``.

    A reply that calls a tool is not an answer to the one-shot question —
    the model went on with the task instead — so it counts as empty too.
    """
    block_types = [part.type for part in message.parts]
    text = "".join(
        part.content or "" for part in message.parts if part.type == "text"
    ).strip()
    if message.function_call_parts:
        raise EmptyCompletionError("tool_use", block_types)
    if not text:
        raise EmptyCompletionError(None, block_types)
    return text


class BaseProvider(ABC):
    # Whether the provider can run Anthropic's server-side tool search tool
    # (see AnthropicProvider). enable_tool_search() checks this before
    # switching to server-side mode.
    supports_server_tool_search = False

    def prepare_messages(
        self,
        transform_function: typing.Callable,
        transform_list_function: typing.Callable = lambda x: x,
        extra_messages: typing.Sequence[Message] = (),
    ) -> list[dict]:
        """Prepare messages for API requests using a transformation function.

        Context and instruction are not included here — each provider adds
        them to the system prompt natively (e.g. Anthropic's ``system``
        field, OpenAI's system messages).

        ``user_context`` (untrusted, user-provided content) is prepended to
        the FIRST user message of the conversation so the model sees it as
        user input, not as system instructions.

        Position matters for prompt caching: caching is a prefix match, so a
        block that moves invalidates every message after its old position.
        Anchoring it to the first user message keeps it in a slot that never
        moves — across the iterations of a tool loop *and* across turns —
        instead of hopping to each new human message and re-writing the
        cached prefix from the previous turn onward.  A system block would be
        equally stable but would give untrusted content system authority,
        which is exactly what this indirection avoids.

        ``extra_messages`` are appended after the conversation for this
        request only, without entering ``chat.messages``.
        """
        all_messages = self.chat.examples + self.chat.messages

        # Prepend user_context to the first user message.  Build a new
        # Message to avoid mutating the original (prepare_messages is
        # called on every LLM round-trip within a tool loop).  Examples are
        # few-shot templates, so only real conversation messages qualify.
        if self.chat.user_context and self.chat.messages:
            offset = len(self.chat.examples)
            first_user_idx = None
            for i, message in enumerate(self.chat.messages):
                if message.role == "user":
                    first_user_idx = offset + i
                    break

            if first_user_idx is not None:
                orig = all_messages[first_user_idx]
                context_part = MessagePart(type="text", content=self.chat.user_context)
                patched = Message(
                    role=orig.role,
                    name=orig.name,
                    id=orig.id,
                    parts=[context_part, *orig.parts],
                )
                # Rebuilding a Message drops every flag not passed to the
                # constructor; is_live decides whether thinking blocks are
                # replayed, so it has to travel with the parts.
                patched.is_live = orig.is_live
                all_messages = (
                    all_messages[:first_user_idx]
                    + [patched]
                    + all_messages[first_user_idx + 1 :]
                )

        messages = all_messages + list(extra_messages)
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

    async def complete(
        self,
        messages: list[dict],
        system: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """One-shot text completion outside the conversation loop.

        Used for auxiliary LLM calls such as compaction summaries.
        ``messages`` are simple ``{"role", "content"}`` dicts; ``model``
        defaults to the provider's configured model.  Returns the response
        text; raises ``EmptyCompletionError`` when the response has none.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support one-shot completions. "
            "Please implement complete() to enable features like compaction."
        )

    async def complete_conversation(self, prompt: str, max_tokens: int = 4096) -> str:
        """Ask one question about the conversation, from the prompt cache.

        Sends the request the conversation loop would send next — same
        model, tools, system, thinking/effort, tool_choice and cache
        breakpoints — with one extra user message holding ``prompt``, so the
        whole conversation is read from the provider's prefix cache instead
        of being paid again. Neither the prompt nor the reply enters
        ``chat.messages``.  Returns the reply text; raises
        ``EmptyCompletionError`` when the reply has no text or calls a tool,
        and ``NotImplementedError`` when the provider cannot replay its
        request (callers then fall back to ``complete()``).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot replay the conversation request."
        )

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
