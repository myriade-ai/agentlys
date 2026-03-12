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


class TestCompactionRendering(unittest.TestCase):
    """Tests for rendering compaction parts in to_markdown() and to_terminal()."""

    def test_to_markdown_includes_compaction(self):
        msg = Message(
            role="user",
            parts=[MessagePart(type="compaction", content="Summary of conversation")],
        )
        md = msg.to_markdown()
        self.assertIn("[Previous conversation summary]", md)
        self.assertIn("Summary of conversation", md)

    def test_to_markdown_mixed_compaction_and_text(self):
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="compaction", content="Earlier discussion summary"),
                MessagePart(type="text", content="New question here"),
            ],
        )
        md = msg.to_markdown()
        self.assertIn("Earlier discussion summary", md)
        self.assertIn("New question here", md)

    def test_to_terminal_includes_compaction(self):
        msg = Message(
            role="user",
            parts=[MessagePart(type="compaction", content="Terminal summary test")],
        )
        term = msg.to_terminal()
        self.assertIn("[Previous conversation summary]", term)
        self.assertIn("Terminal summary test", term)


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


class TestMessageUsage(unittest.TestCase):
    """Tests for Message.usage tracking."""

    def test_usage_default_none(self):
        msg = Message(role="user", content="Hello")
        self.assertIsNone(msg.usage)

    def test_usage_set_on_construction(self):
        msg = Message(
            role="assistant",
            content="Hi",
            usage={"input_tokens": 500, "output_tokens": 100},
        )
        self.assertEqual(msg.usage["input_tokens"], 500)
        self.assertEqual(msg.usage["output_tokens"], 100)

    def test_usage_set_after_construction(self):
        msg = Message(role="assistant", content="Hi")
        msg.usage = {"input_tokens": 1000, "output_tokens": 200}
        self.assertEqual(msg.usage["input_tokens"], 1000)

    def test_usage_set_from_anthropic_provider(self):
        """Verify fetch_async extracts usage from the API response."""
        mock_client = MagicMock()
        agent = Agentlys(instruction="Test", provider=APIProvider.ANTHROPIC)
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = mock_client

        class FakeResponse:
            def to_dict(self):
                return {
                    "role": "assistant",
                    "content": "Hi there",
                    "usage": {"input_tokens": 42000, "output_tokens": 500},
                }

        mock_create = AsyncMock(return_value=FakeResponse())
        with patch.object(agent.provider.client.messages, "create", mock_create):
            loop = asyncio.new_event_loop()
            try:
                response = loop.run_until_complete(agent.ask_async("Hello"))
            finally:
                loop.close()

        self.assertIsNotNone(response.usage)
        self.assertEqual(response.usage["input_tokens"], 42000)
        self.assertEqual(response.usage["output_tokens"], 500)


class TestTokenThresholdCompactionShouldCompact(unittest.TestCase):
    """Tests for TokenThresholdCompaction.should_compact using message usage."""

    def test_should_compact_true_when_over_threshold(self):
        compaction = TokenThresholdCompaction(token_threshold=1000)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content="Hi",
                usage={"input_tokens": 1500, "output_tokens": 100},
            ),
        ]

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
        agent.messages = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content="Hi",
                usage={"input_tokens": 500, "output_tokens": 50},
            ),
        ]

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(compaction.should_compact(agent))
        finally:
            loop.close()

        self.assertFalse(result)

    def test_should_compact_false_when_no_usage(self):
        """When no message has usage data (e.g. first call), return False."""
        compaction = TokenThresholdCompaction(token_threshold=1000)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [Message(role="user", content="Hello")]

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(compaction.should_compact(agent))
        finally:
            loop.close()

        self.assertFalse(result)

    def test_should_compact_uses_most_recent_usage(self):
        """should_compact checks the most recent message with usage, not the first."""
        compaction = TokenThresholdCompaction(token_threshold=5000)
        agent = Agentlys(
            instruction="Test", provider=APIProvider.ANTHROPIC, compaction=compaction
        )
        agent.messages = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content="Hi",
                usage={"input_tokens": 1000, "output_tokens": 50},
            ),
            Message(role="user", content="More context..."),
            Message(
                role="assistant",
                content="Response",
                usage={"input_tokens": 8000, "output_tokens": 200},
            ),
        ]

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

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "<summary>Conversation summary here</summary>"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        # compact() reuses provider.client, so mock that instead of AsyncAnthropic
        agent.provider.client = MagicMock()
        agent.provider.client.messages.create = AsyncMock(return_value=mock_response)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(compaction.compact(agent))
        finally:
            loop.close()

        # Should have: 1 compaction message + 2 preserved messages
        self.assertEqual(len(agent.messages), 3)
        self.assertTrue(agent.messages[0].has_compaction)
        self.assertEqual(
            agent.messages[0].parts[0].content, "Conversation summary here"
        )
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

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Plain summary without tags"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        agent.provider.client = MagicMock()
        agent.provider.client.messages.create = AsyncMock(return_value=mock_response)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(compaction.compact(agent))
        finally:
            loop.close()

        self.assertEqual(
            agent.messages[0].parts[0].content, "Plain summary without tags"
        )

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

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "<summary>Code preserved</summary>"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        mock_create = AsyncMock(return_value=mock_response)
        agent.provider.client = MagicMock()
        agent.provider.client.messages.create = mock_create

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(compaction.compact(agent))
        finally:
            loop.close()

        # Verify custom prompt was used
        call_args = mock_create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        self.assertIn(custom_prompt, user_content)
        self.assertNotIn(DEFAULT_COMPACTION_PROMPT, user_content)


class TestCompactionIntegrationWithAskAsync(unittest.TestCase):
    """Tests that compaction is triggered in ask_async when configured."""

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def _make_fake_response(self, content="Hi there", input_tokens=100):
        class FakeResponse:
            def to_dict(self_inner):
                return {
                    "role": "assistant",
                    "content": content,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 50,
                    },
                }

        return FakeResponse()

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

        mock_create = AsyncMock(return_value=self._make_fake_response())
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

        mock_create = AsyncMock(return_value=self._make_fake_response())
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

        mock_create = AsyncMock(return_value=self._make_fake_response())
        with patch.object(agent.provider.client.messages, "create", mock_create):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(agent.ask_async("Hello"))
            finally:
                loop.close()

        self.assertEqual(result.content, "Hi there")

    def test_usage_available_on_response_message(self):
        """After ask_async, the response Message should carry usage data."""
        agent = Agentlys(
            instruction="Test",
            provider=APIProvider.ANTHROPIC,
        )
        agent.messages = [Message(role="user", content="Hello")]
        agent.provider.client = self.mock_anthropic_client

        mock_create = AsyncMock(
            return_value=self._make_fake_response(input_tokens=42000)
        )
        with patch.object(agent.provider.client.messages, "create", mock_create):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(agent.ask_async("Hello"))
            finally:
                loop.close()

        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage["input_tokens"], 42000)


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
                        parts=[
                            MessagePart(type="compaction", content="Custom summary")
                        ],
                    )
                ] + chat.messages[-2:]

        compactor = MyCompactor()
        self.assertIsInstance(compactor, CompactionHandler)


if __name__ == "__main__":
    unittest.main()
