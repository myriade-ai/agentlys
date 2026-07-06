from PIL import Image as PILImage

from agentlys.model import Image, Message, MessagePart


def test_content_setter_targets_text_part():
    """Setting content must write to the text part, not blindly to parts[0]."""
    message = Message(
        role="assistant",
        parts=[
            MessagePart(type="thinking", thinking="reasoning"),
            MessagePart(type="text", content="hello"),
        ],
    )
    message.content = "world"
    assert message.content == "world"
    assert message.parts[0].thinking == "reasoning"
    assert message.parts[0].content is None


def test_image_to_base64_is_cached():
    pil = PILImage.new("RGB", (4, 4))
    pil.format = "PNG"
    img = Image(pil)

    save_calls = 0
    original_save = pil.save

    def counting_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        return original_save(*args, **kwargs)

    pil.save = counting_save
    first = img.to_base64()
    second = img.to_base64()
    assert first == second
    assert save_calls == 1


def test_image_resize_invalidates_base64_cache():
    pil = PILImage.new("RGB", (4, 4), color="red")
    pil.format = "PNG"
    img = Image(pil)

    before = img.to_base64()
    img.resize((2, 2))
    after = img.to_base64()
    assert before != after
