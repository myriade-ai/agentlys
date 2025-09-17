import pytest
from agentlys import Agentlys, APIProvider


def function_returning_none():
    """A simple function that returns None."""
    print("Function called - returning None")
    return None


@pytest.mark.vcr
def test_anthropic_none_return():
    """Test that a function returning None doesn't cause a crash with Anthropic."""
    agent = Agentlys(
        instruction="You are a helpful assistant that can call functions.",
        provider=APIProvider.ANTHROPIC,
    )
    agent.add_function(function_returning_none)

    user_query = "Call the function_returning_none function"
    messages = []
    for message in agent.run_conversation(user_query):
        # Expected behavior: empty string content for function result message
        if message.role == "function":
            assert message.content == "", (
                "Function result should be an empty string when function returns None"
            )

        print(message.to_markdown())  # For debugging
        messages.append(message)

    # Anthropic models typically add a final message after function call
    # so we expect at least 3 messages (user, assistant with function call, function result)
    # and potentially a 4th message (assistant final response)
    assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}"

    # Verify the function was actually called
    assert any(m.role == "function" for m in messages), (
        "Function message not found in conversation"
    )


@pytest.mark.vcr
def test_openai_none_return():
    """Test that a function returning None doesn't cause a crash with OpenAI."""
    agent = Agentlys(
        instruction="You are a helpful assistant that can call functions.",
        model="gpt-4o",
        provider=APIProvider.OPENAI,
    )
    agent.add_function(function_returning_none)

    user_query = "Call the function_returning_none function"
    messages = []
    for message in agent.run_conversation(user_query):
        # Expected behavior: empty string content for function result message
        if message.role == "function":
            assert message.content == "", (
                "Function result should be an empty string when function returns None"
            )

        print(message.to_markdown())  # For debugging
        messages.append(message)

    # Verify we get exactly 3 messages (user, assistant with function call, function result)
    assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}"

    # Verify the function was actually called
    assert any(m.role == "function" for m in messages), (
        "Function message not found in conversation"
    )
