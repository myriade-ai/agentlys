import os

import pytest
from agentlys import Agentlys, APIProvider, Message
from PIL import Image

img_path = os.path.join(os.path.dirname(__file__), "images", "mileage.jpg")


def report_mileage(km: int):
    """Report the mileage of the car (in km)"""
    print(f"The car has {km} km on the odometer.")
    return f"The car has {km} km on the odometer."


TEST_CASES = [
    APIProvider.ANTHROPIC,
    APIProvider.OPENAI,
    APIProvider.OPENAI_FUNCTION_LEGACY,
]


@pytest.mark.parametrize("provider", TEST_CASES)
@pytest.mark.vcr
def test_read_image(provider):
    agent = Agentlys(provider=provider, use_tools_only=True)
    agent.add_function(report_mileage)

    image = Image.open(img_path)
    message = Message(
        role="user",
        content="report the  mileage of the car",
        image=image,
    )
    response = agent.ask(
        message,
    )
    print(response.to_markdown())
    assert response.function_call["name"] == "report_mileage"
    assert isinstance(response.function_call["arguments"].get("km"), int)
