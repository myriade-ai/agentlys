from unittest.mock import patch

import pytest
from agentlys import Agentlys
from agentlys.model import Message
from agentlys.providers.base_provider import APIProvider


async def async_calculator(**kwargs):
    """Calculate the sum of two numbers."""
    a = kwargs.get("a", 0)
    b = kwargs.get("b", 0)
    return str(a + b)


@pytest.mark.asyncio
async def test_fetch_stream_async_not_implemented_raises_error():
    """Test that providers without streaming support raise NotImplementedError."""
    chat = Agentlys(provider=APIProvider.OPENAI)
    chat.messages.append(Message(role="user", content="Hello"))

    with pytest.raises(NotImplementedError) as exc_info:
        async for _ in chat.provider.fetch_stream_async():
            pass

    assert "does not support streaming" in str(exc_info.value)
    assert "OpenAIProvider" in str(exc_info.value)


async def mock_stream_chunks():
    """Mock async generator that yields streaming chunks."""
    yield {"type": "text", "content": "Hello"}
    yield {"type": "text", "content": " World"}
    yield {
        "type": "message",
        "message": Message(role="assistant", content="Hello World"),
    }


@pytest.mark.asyncio
async def test_ask_stream_async():
    """Test the async streaming version of ask method."""
    chat = Agentlys(provider=APIProvider.ANTHROPIC)

    with patch.object(
        chat.provider, "fetch_stream_async", return_value=mock_stream_chunks()
    ):
        chunks = []
        final_message = None
        async for chunk in chat.ask_stream_async("Say hello"):
            chunks.append(chunk)
            if chunk["type"] == "message":
                final_message = chunk["message"]

        # Should have text chunks and one message
        assert len(chunks) == 3
        text_chunks = [c for c in chunks if c["type"] == "text"]
        assert len(text_chunks) == 2
        assert text_chunks[0]["content"] == "Hello"
        assert text_chunks[1]["content"] == " World"

        assert final_message is not None
        assert isinstance(final_message, Message)
        assert final_message.content == "Hello World"
        # Message should be appended to chat history
        assert len(chat.messages) == 2


@pytest.mark.asyncio
async def test_ask_stream_async_raises_if_no_final_message():
    """Test that ask_stream_async raises error if stream ends without message."""
    chat = Agentlys(provider=APIProvider.ANTHROPIC)

    async def mock_incomplete_stream():
        yield {"type": "text", "content": "Hello"}

    with patch.object(
        chat.provider, "fetch_stream_async", return_value=mock_incomplete_stream()
    ):
        with pytest.raises(RuntimeError) as exc_info:
            async for _ in chat.ask_stream_async("Test"):
                pass

        assert "Stream ended without final message" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_conversation_stream_async():
    """Test streaming conversation without function calls."""
    chat = Agentlys(provider=APIProvider.ANTHROPIC)

    with patch.object(
        chat.provider, "fetch_stream_async", return_value=mock_stream_chunks()
    ):
        events = []
        async for event in chat.run_conversation_stream_async("Say hi"):
            events.append(event)

        # Should have user message, text chunks, and assistant message
        user_events = [e for e in events if e.get("type") == "user"]
        text_events = [e for e in events if e.get("type") == "text"]
        assistant_events = [e for e in events if e.get("type") == "assistant"]

        assert len(user_events) == 1
        assert user_events[0]["message"].content == "Say hi"
        assert len(text_events) == 2
        assert len(assistant_events) == 1


async def mock_stream_with_function_call():
    """Mock async generator that yields a function call response."""
    yield {"type": "text", "content": "I'll calculate that."}
    yield {
        "type": "message",
        "message": Message(
            role="assistant",
            content="I'll calculate that.",
            function_call={"name": "async_calculator", "arguments": {"a": 2, "b": 3}},
            function_call_id="call_123",
        ),
    }


async def mock_stream_final_response():
    """Mock async generator for final response after function call."""
    yield {"type": "text", "content": "The result is 5."}
    yield {
        "type": "message",
        "message": Message(role="assistant", content="The result is 5."),
    }


@pytest.mark.asyncio
async def test_run_conversation_stream_async_with_function():
    """Test streaming conversation with function calls."""
    chat = Agentlys(provider=APIProvider.ANTHROPIC)
    chat.add_function(async_calculator)

    call_count = 0

    def get_mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_stream_with_function_call()
        else:
            return mock_stream_final_response()

    with patch.object(chat.provider, "fetch_stream_async", side_effect=get_mock_stream):
        events = []
        async for event in chat.run_conversation_stream_async("Calculate 2 + 3"):
            events.append(event)

        # Should have user, text chunks, assistant, function, more text, and final assistant
        user_events = [e for e in events if e.get("type") == "user"]
        function_events = [e for e in events if e.get("type") == "function"]
        assistant_events = [e for e in events if e.get("type") == "assistant"]
        text_events = [e for e in events if e.get("type") == "text"]

        assert len(user_events) == 1
        assert len(function_events) == 1
        assert function_events[0]["message"].content == "5"  # Calculator result
        assert len(assistant_events) == 2  # One for function call, one for final
        assert len(text_events) >= 2  # At least 2 text chunks


@pytest.mark.asyncio
async def test_run_conversation_stream_async_raises_if_no_response():
    """Test that run_conversation_stream_async raises if no response."""
    chat = Agentlys(provider=APIProvider.ANTHROPIC)

    async def mock_incomplete_stream():
        yield {"type": "text", "content": "Hello"}

    with patch.object(
        chat.provider, "fetch_stream_async", return_value=mock_incomplete_stream()
    ):
        with pytest.raises(RuntimeError) as exc_info:
            async for _ in chat.run_conversation_stream_async("Test"):
                pass

        assert "Stream ended without" in str(exc_info.value)
