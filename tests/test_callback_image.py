"""We want to be able to return an image from a function call."""

import os

import pytest
from agentlys import Agentlys, APIProvider
from PIL import Image

img_path = os.path.join(os.path.dirname(__file__), "images", "mileage.jpg")


def read_image():
    return Image.open(img_path)


@pytest.mark.parametrize(
    "provider",
    [
        APIProvider.OPENAI_FUNCTION_LEGACY,
        APIProvider.ANTHROPIC,
    ],
)
@pytest.mark.vcr
def test_anthropic_callback_image(provider):
    agent = Agentlys(
        provider=provider,
    )
    agent.add_function(read_image)

    user_query = "Hello, read the image and explain it"
    messages = []
    for message in agent.run_conversation(user_query):
        messages.append(message)
