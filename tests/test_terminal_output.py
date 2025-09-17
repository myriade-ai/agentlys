#!/usr/bin/env python
"""
Test script for terminal output with clickable image links.
Run this script in your terminal to test the clickable links.
"""

import os

from PIL import Image as PILImage

from agentlys.model import Message, MessagePart


def main(display_image: bool):
    print("Testing terminal output with clickable links...\n")

    # Read test image
    img_path = os.path.join(os.path.dirname(__file__), "images", "mileage.jpg")
    test_image = PILImage.open(img_path)

    # Create a user message with text
    user_msg = Message(role="user", content="Here's a test message with some text.")
    print(user_msg.to_terminal())
    print("\n" + "-" * 50 + "\n")

    # Create an assistant message with text and an image
    assistant_msg = Message(
        role="assistant", content="I'm responding with an image.", parts=[]
    )
    assistant_msg.parts.append(MessagePart(type="image", image=test_image))
    print(assistant_msg.to_terminal(display_image=args.display_image))
    print("\n" + "-" * 50 + "\n")

    # Create a function result message with image
    function_msg = Message(
        role="function", image=test_image, function_call_id="test_function_1"
    )
    print(function_msg.to_terminal(display_image=args.display_image))
    print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display-image",
        action="store_true",
        help="Display images inline in supported terminals",
    )
    args = parser.parse_args()
    main(args)
