"""
This is a demo of how to use the openai_function_shim provider,
which is a hacky way to use function calling with o1 OpenAI models.

It is not recommended for production use, but it can be useful for
testing and experimentation.
"""

import pytest
from agentlys import Agentlys, APIProvider


def multiply(a: int, b: int) -> int:
    return a * b


@pytest.mark.vcr
def test_openai_function_shim():
    # The only model thing you need to do is set the provider to "openai_function_shim"
    agent = Agentlys(provider=APIProvider.OPENAI_FUNCTION_SHIM, model="o1-preview")
    agent.add_function(multiply)

    text = "What is 343354 * 13243343214"
    for message in agent.run_conversation(text):
        print(message.to_markdown())


if __name__ == "__main__":
    test_openai_function_shim()
