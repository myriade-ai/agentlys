import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentlys import Agentlys, APIProvider, Message


class TestAnthropic(unittest.TestCase):
    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def test_transform_conversation_anthropic(self):
        agent = Agentlys(instruction="Test instruction", provider=APIProvider.ANTHROPIC)
        agent.messages = [
            Message(
                role="function",
                name="SUBMIT",
                content="",
                function_call_id="example_16",
            ),
            Message(role="user", content="Plot distribution of stations per city"),
        ]

        expected_output = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "example_16", "content": ""},
                    {
                        "type": "text",
                        "text": "Plot distribution of stations per city",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ]
        agent.provider.client = self.mock_anthropic_client

        class FakeAnthropicMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

            def to_dict(self):
                return {
                    "role": self.role,
                    "content": self.content,
                }

        return_value = FakeAnthropicMessage(
            role="assistant",
            content="test",
        )

        mock_create = AsyncMock(return_value=return_value)
        with patch.object(agent.provider.client.messages, "create", mock_create):
            import asyncio

            asyncio.get_event_loop().run_until_complete(agent.provider.fetch_async())
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            actual_messages = call_args.kwargs["messages"]

            self.assertEqual(actual_messages, expected_output)


class TestStripThinkingFromPriorTurns(unittest.TestCase):
    """Tests for AnthropicProvider._strip_thinking_from_prior_turns."""

    def _call(self, messages):
        from agentlys.providers.anthropic import AnthropicProvider

        return AnthropicProvider._strip_thinking_from_prior_turns(messages)

    def test_strips_thinking_from_all_assistants_without_tool_loop(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "old thought",
                        "signature": "sig1",
                    },
                    {"type": "text", "text": "response 1"},
                ],
            },
            {"role": "user", "content": "follow up"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "new thought",
                        "signature": "sig2",
                    },
                    {"type": "text", "text": "response 2"},
                ],
            },
        ]
        result = self._call(messages)
        # Both assistants stripped — no tool_result follows the last one
        self.assertEqual(
            result[1]["content"],
            [{"type": "text", "text": "response 1"}],
        )
        self.assertEqual(
            result[3]["content"],
            [{"type": "text", "text": "response 2"}],
        )

    def test_strips_last_assistant_thinking_when_no_tool_loop(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "my thought", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]
        result = self._call(messages)
        # Last assistant but no tool_result follows — thinking stripped
        self.assertEqual(
            result[1]["content"],
            [{"type": "text", "text": "answer"}],
        )

    def test_preserves_last_assistant_thinking_in_tool_loop(self):
        messages = [
            {"role": "user", "content": "query the database"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "thought", "signature": "sig"},
                    {"type": "text", "text": "I'll run a query"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "query",
                        "input": {"sql": "SELECT 1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "1"},
                ],
            },
        ]
        result = self._call(messages)
        # Last assistant followed by tool_result — thinking preserved
        self.assertEqual(result[1]["content"], messages[1]["content"])

    def test_non_thinking_blocks_untouched(self):
        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll use a tool"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "query",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        ]
        result = self._call(messages)
        # First assistant has no thinking, should be unchanged
        self.assertEqual(result[1]["content"], messages[1]["content"])
        # User message unchanged
        self.assertEqual(result[2], messages[2])
        # Last assistant unchanged
        self.assertEqual(result[3], messages[3])

    def test_user_and_function_messages_untouched(self):
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "data"},
                ],
            },
        ]
        result = self._call(messages)
        self.assertEqual(result, messages)

    def test_strips_redacted_thinking(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {"type": "text", "text": "response"},
                ],
            },
            {"role": "user", "content": "more"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "final"}],
            },
        ]
        result = self._call(messages)
        # First assistant: redacted_thinking stripped
        self.assertEqual(
            result[1]["content"],
            [{"type": "text", "text": "response"}],
        )

    def test_no_assistant_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
        ]
        result = self._call(messages)
        self.assertEqual(result, messages)

    def test_empty_messages(self):
        self.assertEqual(self._call([]), [])


class TestCacheControlPlacement(unittest.TestCase):
    """Tests that cache_control is placed on the last system block."""

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def _make_agent_and_call(self, instruction, initial_tools_states=None):
        agent = Agentlys(instruction=instruction, provider=APIProvider.ANTHROPIC)
        agent.messages = [Message(role="user", content="hello")]
        if initial_tools_states is not None:
            agent._initial_tools_states = initial_tools_states
        agent.provider.client = self.mock_anthropic_client

        class FakeAnthropicMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        mock_create = AsyncMock(
            return_value=FakeAnthropicMessage(role="assistant", content="test")
        )
        with patch.object(agent.provider.client.messages, "create", mock_create):
            import asyncio

            asyncio.get_event_loop().run_until_complete(agent.provider.fetch_async())

        return mock_create.call_args.kwargs

    def test_cache_control_on_last_system_block(self):
        """When system has instruction + tool states, cache_control should be on system[-1]."""
        kwargs = self._make_agent_and_call(
            instruction="You are a data analyst.",
            initial_tools_states="Tables: users, orders",
        )
        system = kwargs["system"]
        self.assertEqual(len(system), 2)
        # system[0] (instruction) should NOT have cache_control
        self.assertNotIn("cache_control", system[0])
        # system[-1] (tool states) should have cache_control
        self.assertEqual(system[-1]["cache_control"], {"type": "ephemeral"})

    def test_cache_control_on_sole_system_block(self):
        """When system has only instruction (no tool states), cache_control should be on system[0]."""
        kwargs = self._make_agent_and_call(
            instruction="You are a data analyst.",
            initial_tools_states=None,
        )
        system = kwargs["system"]
        self.assertEqual(len(system), 1)
        # The sole system block should have cache_control
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})


if __name__ == "__main__":
    unittest.main()
