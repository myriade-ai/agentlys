import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentlys import Agentlys, APIProvider, Message, MessagePart
from agentlys.compaction import (
    DEFAULT_COMPACTION_PROMPT,
    CompactionHandler,
    TokenThresholdCompaction,
)


class TestCompactionMessagePart(unittest.TestCase):
    """Tests for the compaction MessagePart type."""

    def test_create_compaction_part(self):
        part = MessagePart(type="compaction", content="Summary of conversation")
        self.assertEqual(part.type, "compaction")
        self.assertEqual(part.content, "Summary of conversation")

    def test_message_has_compaction_true(self):
        msg = Message(
            role="user",
            parts=[MessagePart(type="compaction", content="Summary")],
        )
        self.assertTrue(msg.has_compaction)

    def test_message_has_compaction_false(self):
        msg = Message(role="user", content="Hello")
        self.assertFalse(msg.has_compaction)

    def test_from_anthropic_dict_parses_compaction(self):
        msg = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "compaction", "content": "Summary of previous conversation"},
                {"type": "text", "text": "Continuing from where we left off..."},
            ],
        )
        self.assertEqual(len(msg.parts), 2)
        self.assertEqual(msg.parts[0].type, "compaction")
        self.assertEqual(msg.parts[0].content, "Summary of previous conversation")
        self.assertEqual(msg.parts[1].type, "text")

    def test_from_anthropic_dict_compaction_empty_content(self):
        msg = Message.from_anthropic_dict(
            role="assistant",
            content=[{"type": "compaction"}],
        )
        self.assertEqual(len(msg.parts), 1)
        self.assertEqual(msg.parts[0].type, "compaction")
        self.assertEqual(msg.parts[0].content, "")


class TestCompactionSerialization(unittest.TestCase):
    """Tests for serializing compaction parts to Anthropic API format."""

    def test_part_to_anthropic_dict_compaction(self):
        from agentlys.providers.anthropic import part_to_anthropic_dict

        part = MessagePart(type="compaction", content="Summary text here")
        result = part_to_anthropic_dict(part)
        self.assertEqual(result["type"], "text")
        self.assertIn("[Previous conversation summary]", result["text"])
        self.assertIn("Summary text here", result["text"])

    def test_message_to_anthropic_dict_with_compaction(self):
        from agentlys.providers.anthropic import message_to_anthropic_dict

        msg = Message(
            role="user",
            parts=[MessagePart(type="compaction", content="Conversation summary")],
        )
        result = message_to_anthropic_dict(msg)
        self.assertEqual(result["role"], "user")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("Conversation summary", result["content"][0]["text"])


class TestTokenThresholdCompactionShouldCompact(unittest.TestCase):
    """Tests for TokenThresholdCompaction.should_compact."""

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def test_should_compact_true_when_over_threshold(self):
        compaction = TokenThresholdCompaction(token_threshold=1000)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = self.mock_anthropic_client

        # Mock count_tokens to return a value above threshold
        mock_count_result = MagicMock()
        mock_count_result.input_tokens = 1500

        mock_count = AsyncMock(return_value=mock_count_result)

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.messages.count_tokens = mock_count
            mock_client_cls.return_value = mock_instance

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(compaction.should_compact(agent))
            finally:
                loop.close()

        self.assertTrue(result)

    def test_should_compact_false_when_under_threshold(self):
        compaction = TokenThresholdCompaction(token_threshold=1000)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = self.mock_anthropic_client

        mock_count_result = MagicMock()
        mock_count_result.input_tokens = 500

        mock_count = AsyncMock(return_value=mock_count_result)

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.messages.count_tokens = mock_count
            mock_client_cls.return_value = mock_instance

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(compaction.should_compact(agent))
            finally:
                loop.close()

        self.assertFalse(result)

    def test_should_compact_falls_back_to_message_count(self):
        """When token counting fails, fall back to message count heuristic."""
        compaction = TokenThresholdCompaction(token_threshold=1000)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        # Add enough messages to exceed the 40-message fallback threshold
        agent.messages = [
            Message(role="user", content=f"Message {i}") for i in range(45)
        ]
        agent.provider.client = self.mock_anthropic_client

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.messages.count_tokens = AsyncMock(
                side_effect=Exception("API error")
            )
            mock_client_cls.return_value = mock_instance

            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(compaction.should_compact(agent))
            finally:
                loop.close()

        self.assertTrue(result)


class TestTokenThresholdCompactionCompact(unittest.TestCase):
    """Tests for TokenThresholdCompaction.compact."""

    def test_compact_replaces_older_messages_with_summary(self):
        compaction = TokenThresholdCompaction(preserve_last_n=2)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [
            Message(role="user", content="First message"),
            Message(role="assistant", content="First response"),
            Message(role="user", content="Second message"),
            Message(role="assistant", content="Second response"),
            Message(role="user", content="Third message"),
            Message(role="assistant", content="Third response"),
        ]

        # Mock the summary API call
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="<summary>Conversation summary here</summary>")]

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_instance

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(compaction.compact(agent))
            finally:
                loop.close()

        # Should have: 1 compaction message + 2 preserved messages
        self.assertEqual(len(agent.messages), 3)
        self.assertTrue(agent.messages[0].has_compaction)
        self.assertEqual(agent.messages[0].parts[0].content, "Conversation summary here")
        self.assertEqual(agent.messages[1].content, "Third message")
        self.assertEqual(agent.messages[2].content, "Third response")

    def test_compact_skips_when_too_few_messages(self):
        compaction = TokenThresholdCompaction(preserve_last_n=4)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        original_messages = list(agent.messages)

        with patch("anthropic.AsyncAnthropic"):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(compaction.compact(agent))
            finally:
                loop.close()

        # Messages should be unchanged
        self.assertEqual(len(agent.messages), len(original_messages))

    def test_compact_extracts_summary_without_tags(self):
        """When the model doesn't use summary tags, use the full response."""
        compaction = TokenThresholdCompaction(preserve_last_n=2)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [
            Message(role="user", content="First"),
            Message(role="assistant", content="Response 1"),
            Message(role="user", content="Second"),
            Message(role="assistant", content="Response 2"),
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Plain summary without tags")]

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_instance

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(compaction.compact(agent))
            finally:
                loop.close()

        self.assertEqual(agent.messages[0].parts[0].content, "Plain summary without tags")

    def test_compact_uses_custom_instructions(self):
        custom_prompt = "Preserve all code snippets verbatim."
        compaction = TokenThresholdCompaction(
            preserve_last_n=2, instructions=custom_prompt
        )
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [
            Message(role="user", content="First"),
            Message(role="assistant", content="Response"),
            Message(role="user", content="Second"),
            Message(role="assistant", content="Response 2"),
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="<summary>Code preserved</summary>")]

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_instance

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(compaction.compact(agent))
            finally:
                loop.close()

            # Verify custom prompt was used
            call_args = mock_instance.messages.create.call_args
            user_content = call_args.kwargs["messages"][0]["content"]
            self.assertIn(custom_prompt, user_content)
            self.assertNotIn(DEFAULT_COMPACTION_PROMPT, user_content)


class TestCompactionIntegrationWithAskAsync(unittest.TestCase):
    """Tests that compaction is triggered in ask_async when configured."""

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def test_compaction_triggered_in_ask_async(self):
        """When should_compact returns True, compact is called before the LLM call."""
        mock_compaction = MagicMock()
        mock_compaction.should_compact = AsyncMock(return_value=True)
        mock_compaction.compact = AsyncMock()

        agent = Agentlys(
            instruction="Test",
            provider=APIProvider.ANTHROPIC,
            compaction=mock_compaction,
        )
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = self.mock_anthropic_client

        class FakeAnthropicMessage:
            def __init__(self):
                self.role = "assistant"
                self.content = "Hi there"

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        mock_create = AsyncMock(return_value=FakeAnthropicMessage())
        with patch.object(agent.provider.client.messages, "create", mock_create):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(agent.ask_async("Hello"))
            finally:
                loop.close()

        mock_compaction.should_compact.assert_called_once_with(agent)
        mock_compaction.compact.assert_called_once_with(agent)

    def test_compaction_not_triggered_when_under_threshold(self):
        """When should_compact returns False, compact is NOT called."""
        mock_compaction = MagicMock()
        mock_compaction.should_compact = AsyncMock(return_value=False)
        mock_compaction.compact = AsyncMock()

        agent = Agentlys(
            instruction="Test",
            provider=APIProvider.ANTHROPIC,
            compaction=mock_compaction,
        )
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = self.mock_anthropic_client

        class FakeAnthropicMessage:
            def __init__(self):
                self.role = "assistant"
                self.content = "Hi"

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        mock_create = AsyncMock(return_value=FakeAnthropicMessage())
        with patch.object(agent.provider.client.messages, "create", mock_create):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(agent.ask_async("Hello"))
            finally:
                loop.close()

        mock_compaction.should_compact.assert_called_once()
        mock_compaction.compact.assert_not_called()

    def test_no_compaction_when_not_configured(self):
        """When compaction is None, no compaction check happens."""
        agent = Agentlys(
            instruction="Test",
            provider=APIProvider.ANTHROPIC,
        )
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = self.mock_anthropic_client

        class FakeAnthropicMessage:
            def __init__(self):
                self.role = "assistant"
                self.content = "Hi"

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        mock_create = AsyncMock(return_value=FakeAnthropicMessage())
        with patch.object(agent.provider.client.messages, "create", mock_create):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(agent.ask_async("Hello"))
            finally:
                loop.close()

        # Should succeed without errors
        self.assertEqual(result.content, "Hi")


class TestCustomCompactionHandler(unittest.TestCase):
    """Tests for using a custom CompactionHandler."""

    def test_custom_handler_satisfies_protocol(self):
        class MyCompactor:
            async def should_compact(self, chat):
                return len(chat.messages) > 10

            async def compact(self, chat):
                from agentlys.model import Message, MessagePart

                chat.messages = [
                    Message(
                        role="user",
                        parts=[MessagePart(type="compaction", content="Custom summary")],
                    )
                ] + chat.messages[-2:]

        compactor = MyCompactor()
        self.assertIsInstance(compactor, CompactionHandler)


if __name__ == "__main__":
    unittest.main()
