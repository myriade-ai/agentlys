import asyncio

import pytest
from agentlys import Agentlys
from agentlys.model import Message
from agentlys.providers.base_provider import APIProvider


# Mock async function that simulates an external tool
async def async_calculator(**kwargs):
    a = kwargs.get("a", 0)
    b = kwargs.get("b", 0)
    return str(a + b)


# Mock async function that returns a more complex result
async def async_data_processor(**kwargs):
    await asyncio.sleep(0.1)  # Simulate some async work
    return {"result": "processed data"}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_call_with_signature_async():
    """Test that _call_with_signature properly handles async functions"""
    chat = Agentlys()

    # Test with async function
    result = await chat._call_with_signature(async_calculator, None, a=1, b=2)
    assert result == "3"

    # Test with async function that returns complex data
    result = await chat._call_with_signature(async_data_processor, None)
    assert result == {"result": "processed data"}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ask_async():
    """Test the async version of ask method"""
    chat = Agentlys()

    # Add a test function
    chat.add_function(async_calculator)

    # Test asking a question
    response = await chat.ask_async("What is 1 + 2?")
    assert isinstance(response, Message)
    assert len(chat.messages) == 2  # Question and response


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_run_conversation_async():
    """Test the async generator behavior of run_conversation_async"""
    chat = Agentlys()

    # Add test functions
    chat.add_function(async_calculator)

    messages = []
    async for message in chat.run_conversation_async("Calculate 1 + 2"):
        messages.append(message)
        assert isinstance(message, Message)

    # Should have at least the initial question
    assert len(messages) >= 1
    assert messages[0].content == "Calculate 1 + 2"


async def run_conversation_async_multiple_turns(provider: APIProvider):
    """Test multiple turns in an async conversation"""
    chat = Agentlys(provider=provider)

    # Add both test functions
    chat.add_function(async_calculator)

    chat.add_function(async_data_processor)

    messages = []
    async for message in chat.run_conversation_async(
        "First calculate 1 + 2, then process the result"
    ):
        messages.append(message)
        assert isinstance(message, Message)

    # Should have multiple messages for the multi-turn conversation
    assert (
        len(messages) >= 3
    )  # Initial question + at least one response and function result


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_run_conversation_async_multiple_turns_anthropic():
    """Test multiple turns in an async conversation"""
    await run_conversation_async_multiple_turns(APIProvider.ANTHROPIC)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_run_conversation_async_multiple_turns_openai():
    """Test multiple turns in an async conversation"""
    await run_conversation_async_multiple_turns(APIProvider.OPENAI)


@pytest.mark.vcr
def test_run_conversation_sync_wrapper():
    """Test that the sync wrapper for run_conversation works"""
    chat = Agentlys()

    # Add a test function
    chat.add_function(async_calculator)

    messages = list(chat.run_conversation("Calculate 1 + 2"))

    # Should have at least the initial question
    assert len(messages) >= 1
    assert messages[0].content == "Calculate 1 + 2"
    assert all(isinstance(m, Message) for m in messages)
