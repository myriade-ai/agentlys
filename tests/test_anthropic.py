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


class TestContextMutationInPrepareMessages(unittest.TestCase):
    """Tests that prepare_messages does not mutate the original messages.

    Bug: base_provider.prepare_messages() prepends self.chat.context to
    messages[0].parts[0].content IN PLACE. On every LLM call within a
    tool loop, context is prepended again, causing:
      - Call 1: "CONTEXT\\nHello"
      - Call 2: "CONTEXT\\nCONTEXT\\nHello"
      - Call 3: "CONTEXT\\nCONTEXT\\nCONTEXT\\nHello"
    This invalidates the Anthropic prompt cache because messages[0] changes
    every call, so the entire message prefix hash changes.
    """

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def _make_agent(self, context="## Project\nname: test_db"):
        agent = Agentlys(
            instruction="You are a data analyst.",
            provider=APIProvider.ANTHROPIC,
            context=context,
        )
        agent.provider.client = self.mock_anthropic_client
        return agent

    def _call_prepare(self, agent):
        """Call _prepare_request_params and return the messages kwarg."""

        class FakeAnthropicMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        mock_create = AsyncMock(
            return_value=FakeAnthropicMessage(role="assistant", content="ok")
        )
        with patch.object(agent.provider.client.messages, "create", mock_create):
            import asyncio

            asyncio.get_event_loop().run_until_complete(agent.provider.fetch_async())

        return mock_create.call_args.kwargs["messages"]

    def test_context_not_accumulated_across_calls(self):
        """Calling prepare_messages N times must produce identical messages[0] each time."""
        agent = self._make_agent()
        agent.messages = [
            Message(role="user", content="What tables are available?"),
            Message(role="assistant", content="Let me check."),
            Message(role="user", content="Thanks"),
        ]

        msgs1 = self._call_prepare(agent)
        msgs2 = self._call_prepare(agent)
        msgs3 = self._call_prepare(agent)

        # The first message text must be identical across all 3 calls
        text1 = msgs1[0]["content"][0]["text"]
        text2 = msgs2[0]["content"][0]["text"]
        text3 = msgs3[0]["content"][0]["text"]

        self.assertEqual(text1, text2, "Context was accumulated on 2nd call")
        self.assertEqual(text2, text3, "Context was accumulated on 3rd call")

    def test_original_message_not_mutated(self):
        """The original Message object in agent.messages must not be modified."""
        agent = self._make_agent()
        original_content = "What tables are available?"
        agent.messages = [
            Message(role="user", content=original_content),
        ]

        self._call_prepare(agent)

        # The original message's content must be unchanged
        self.assertEqual(
            agent.messages[0].parts[0].content,
            original_content,
            "prepare_messages mutated the original message in-place",
        )

    def test_context_is_present_in_output(self):
        """Context should still be prepended in the API output (just not mutated in-place)."""
        context = "## Project\nname: test_db"
        agent = self._make_agent(context=context)
        original_content = "Hello"
        agent.messages = [
            Message(role="user", content=original_content),
        ]

        msgs = self._call_prepare(agent)

        first_text = msgs[0]["content"][0]["text"]
        self.assertIn(context, first_text, "Context should be in the API output")
        self.assertIn(
            original_content, first_text, "Original content should be in the API output"
        )


if __name__ == "__main__":
    unittest.main()
