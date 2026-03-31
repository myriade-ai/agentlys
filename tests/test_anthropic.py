import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentlys import Agentlys, APIProvider, Message, MessagePart


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
                    "usage": {"input_tokens": 100, "output_tokens": 50},
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
                return {"role": self.role, "content": self.content, "usage": {"input_tokens": 100, "output_tokens": 50}}

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


class TestContextInSystemPrompt(unittest.TestCase):
    """Tests that context is included in the system prompt, not in user messages."""

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
        """Call fetch_async and return the full kwargs dict."""

        mock_create = AsyncMock(
            return_value=_FakeAnthropicMessage(role="assistant", content="ok")
        )
        with patch.object(agent.provider.client.messages, "create", mock_create):
            import asyncio

            asyncio.get_event_loop().run_until_complete(agent.provider.fetch_async())

        return mock_create.call_args.kwargs

    def test_context_in_system_not_in_user_messages(self):
        """Context must appear in the system field, not in user messages."""
        context = "## Project\nname: test_db"
        agent = self._make_agent(context=context)
        agent.messages = [Message(role="user", content="Hello")]

        kwargs = self._call_prepare(agent)

        # Context should be in system
        system = kwargs["system"]
        system_texts = [b["text"] for b in system]
        self.assertTrue(
            any(context in t for t in system_texts),
            "Context should be in the system field",
        )

        # Context should NOT be in the user message
        first_msg_content = kwargs["messages"][0]["content"]
        if isinstance(first_msg_content, str):
            user_texts = [first_msg_content]
        else:
            user_texts = [b.get("text", "") for b in first_msg_content if isinstance(b, dict)]
        self.assertFalse(
            any(context in t for t in user_texts),
            "Context should NOT be in user messages",
        )

    def test_system_ordering_instruction_context_tools(self):
        """System blocks must be ordered: instruction, context, tool_states."""
        context = "## Project\nname: test_db"
        agent = self._make_agent(context=context)
        agent.messages = [Message(role="user", content="Hello")]

        # Simulate tool states being captured
        agent._initial_tools_states = "## Initial Tools States\n### DummyTool\nA tool"

        kwargs = self._call_prepare(agent)
        system = kwargs["system"]

        # Should have 3 blocks: instruction, context, tool_states
        self.assertEqual(len(system), 3)
        self.assertIn("You are a data analyst.", system[0]["text"])
        self.assertIn(context, system[1]["text"])
        self.assertIn("Initial Tools States", system[2]["text"])

    def test_context_stable_across_calls(self):
        """System field must be identical across repeated calls (cache-safe)."""
        agent = self._make_agent()
        agent.messages = [
            Message(role="user", content="What tables are available?"),
            Message(role="assistant", content="Let me check."),
            Message(role="user", content="Thanks"),
        ]

        kwargs1 = self._call_prepare(agent)
        kwargs2 = self._call_prepare(agent)
        kwargs3 = self._call_prepare(agent)

        self.assertEqual(kwargs1["system"], kwargs2["system"])
        self.assertEqual(kwargs2["system"], kwargs3["system"])

    def test_no_context_omits_block(self):
        """When context is None, system should not include an empty block."""
        agent = self._make_agent(context=None)
        agent.messages = [Message(role="user", content="Hello")]

        kwargs = self._call_prepare(agent)
        system = kwargs["system"]

        # Should only have instruction (no context block, no tool_states)
        self.assertEqual(len(system), 1)
        self.assertIn("You are a data analyst.", system[0]["text"])


class TestUserContext(unittest.TestCase):
    """user_context must be prepended to the last user message, not in system."""

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def _call_prepare(self, agent):
        mock_create = AsyncMock(
            return_value=_FakeAnthropicMessage(role="assistant", content="ok")
        )
        with patch.object(agent.provider.client.messages, "create", mock_create):
            import asyncio

            asyncio.get_event_loop().run_until_complete(agent.provider.fetch_async())
        return mock_create.call_args.kwargs

    def test_user_context_in_user_message_not_system(self):
        """user_context must appear in user messages, not in system."""
        agent = Agentlys(
            instruction="You are a helper.",
            provider=APIProvider.ANTHROPIC,
            user_context="project:\n  name: Sales DB",
        )
        agent.provider.client = self.mock_anthropic_client
        agent.messages = [Message(role="user", content="Hello")]

        kwargs = self._call_prepare(agent)

        # Must NOT be in system
        system_texts = [b["text"] for b in kwargs["system"]]
        self.assertFalse(
            any("Sales DB" in t for t in system_texts),
            "user_context should NOT be in system",
        )

        # Must be in the user message
        msg = kwargs["messages"][0]
        content = msg["content"]
        if isinstance(content, str):
            texts = [content]
        else:
            texts = [b.get("text", "") for b in content if isinstance(b, dict)]
        combined = "\n".join(texts)
        self.assertIn("Sales DB", combined)
        self.assertIn("Hello", combined)

    def test_user_context_not_mutated_across_calls(self):
        """Repeated calls must not accumulate user_context."""
        agent = Agentlys(
            instruction="You are a helper.",
            provider=APIProvider.ANTHROPIC,
            user_context="project:\n  name: Sales DB",
        )
        agent.provider.client = self.mock_anthropic_client
        agent.messages = [Message(role="user", content="Hello")]

        kwargs1 = self._call_prepare(agent)
        kwargs2 = self._call_prepare(agent)

        msgs1 = kwargs1["messages"]
        msgs2 = kwargs2["messages"]
        self.assertEqual(msgs1, msgs2)

    def test_no_user_context_leaves_messages_unchanged(self):
        """When user_context is None, messages stay clean."""
        agent = Agentlys(
            instruction="You are a helper.",
            provider=APIProvider.ANTHROPIC,
        )
        agent.provider.client = self.mock_anthropic_client
        agent.messages = [Message(role="user", content="Hello")]

        kwargs = self._call_prepare(agent)

        msg = kwargs["messages"][0]
        content = msg["content"]
        if isinstance(content, str):
            self.assertEqual(content, "Hello")
        else:
            texts = [b.get("text", "") for b in content if isinstance(b, dict)]
            self.assertEqual(len(texts), 1)
            self.assertEqual(texts[0], "Hello")


class TestCacheBreakpointOnPreviousIteration(unittest.TestCase):
    """Tests that cache_control breakpoints are retained across tool loop iterations.

    Bug: _prepare_request_params placed a single breakpoint on messages[-1].
    In a tool loop, messages[-1] moves on every iteration (2 messages appended
    per round: 1 assistant + 1 tool_result).  The previous breakpoint position
    is lost, so Anthropic cannot find the cached prefix for messages — only
    system + tools get cache hits (~10-14K tokens), while the full message
    history (100K+) is re-cached (cache_creation) every call.

    Fix: add a second message breakpoint on messages[-3], which corresponds
    to messages[-1] from the previous iteration.  This uses 4 of 4 allowed
    breakpoints: system[-1], tools[-1], messages[-3], messages[-1].
    """

    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def _make_agent(self):
        agent = Agentlys(
            instruction="You are a data analyst.",
            provider=APIProvider.ANTHROPIC,
            context="## Database\ntables: users, orders",
        )
        agent.provider.client = self.mock_anthropic_client
        agent.functions_schema = [
            {
                "name": "run_query",
                "description": "Run a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                },
            }
        ]
        agent.functions = {"run_query": lambda sql: sql}
        return agent

    def _call_prepare(self, agent):
        mock_create = AsyncMock(
            return_value=_FakeAnthropicMessage(role="assistant", content="ok")
        )
        with patch.object(agent.provider.client.messages, "create", mock_create):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(agent.provider.fetch_async())
            finally:
                loop.close()
        return mock_create.call_args.kwargs

    @staticmethod
    def _find_message_breakpoints(messages):
        """Return indices of messages that contain a cache_control marker."""
        breakpoints = []
        for i, msg in enumerate(messages):
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    breakpoints.append(i)
                    break
        return breakpoints

    def test_previous_breakpoint_retained(self):
        """After appending 2 messages, Call 1's messages[-1] breakpoint
        must still be present in Call 2 (now at messages[-3])."""
        agent = self._make_agent()

        loaded = [
            Message(role="user", content="Show users"),
            Message(
                role="assistant",
                parts=[
                    MessagePart(type="text", content="Querying."),
                    MessagePart(
                        type="function_call",
                        function_call={
                            "name": "run_query",
                            "arguments": {"sql": "SELECT * FROM users"},
                        },
                        function_call_id="old_1",
                    ),
                ],
            ),
            Message(
                role="function", content="Alice,Bob", function_call_id="old_1"
            ),
            Message(role="assistant", content="Here are the users."),
        ]
        question = Message(role="user", content="Count orders per user")

        # Call 1
        agent.messages = loaded + [question]
        kwargs1 = self._call_prepare(agent)
        bp1 = self._find_message_breakpoints(kwargs1["messages"])
        call1_last_bp = bp1[-1]  # messages[-1] breakpoint

        # Call 2: 2 new messages (assistant + tool_result)
        agent.messages = loaded + [
            question,
            Message(
                role="assistant",
                parts=[
                    MessagePart(type="text", content="Counting."),
                    MessagePart(
                        type="function_call",
                        function_call={
                            "name": "run_query",
                            "arguments": {"sql": "SELECT user_id, COUNT(*) FROM orders GROUP BY 1"},
                        },
                        function_call_id="call_1",
                    ),
                ],
            ),
            Message(
                role="function",
                content="1,10\n2,5",
                function_call_id="call_1",
            ),
        ]
        kwargs2 = self._call_prepare(agent)
        bp2 = self._find_message_breakpoints(kwargs2["messages"])

        self.assertIn(
            call1_last_bp,
            bp2,
            f"Call 2 must retain a breakpoint at index {call1_last_bp} "
            f"(Call 1's messages[-1]).  Got breakpoints at {bp2}.",
        )

    def test_two_message_breakpoints_present(self):
        """Call 2+ should have breakpoints at messages[-3] and messages[-1]."""
        agent = self._make_agent()

        agent.messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="Query something"),
            Message(
                role="assistant",
                parts=[
                    MessagePart(
                        type="function_call",
                        function_call={
                            "name": "run_query",
                            "arguments": {"sql": "SELECT 1"},
                        },
                        function_call_id="tc1",
                    ),
                ],
            ),
            Message(role="function", content="1", function_call_id="tc1"),
        ]
        kwargs = self._call_prepare(agent)
        msgs = kwargs["messages"]
        bp = self._find_message_breakpoints(msgs)

        self.assertIn(len(msgs) - 3, bp, "Should have breakpoint at messages[-3]")
        self.assertIn(len(msgs) - 1, bp, "Should have breakpoint at messages[-1]")

    def test_no_messages_minus_3_when_too_few_messages(self):
        """With fewer than 3 messages, only messages[-1] should have a breakpoint."""
        agent = self._make_agent()
        agent.messages = [Message(role="user", content="Hello")]

        kwargs = self._call_prepare(agent)
        msgs = kwargs["messages"]
        bp = self._find_message_breakpoints(msgs)

        self.assertEqual(bp, [len(msgs) - 1], "Only messages[-1] breakpoint expected")

    def test_three_messages_gets_both_breakpoints(self):
        """With exactly 3 messages (first tool-loop follow-up), both breakpoints should be set."""
        agent = self._make_agent()
        agent.messages = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                parts=[
                    MessagePart(type="text", content="Let me check."),
                    MessagePart(
                        type="function_call",
                        function_call={"name": "run_query", "arguments": {"sql": "SELECT 1"}},
                        function_call_id="call_1",
                    ),
                ],
            ),
            Message(role="function", content="1", function_call_id="call_1"),
        ]

        kwargs = self._call_prepare(agent)
        msgs = kwargs["messages"]
        bp = self._find_message_breakpoints(msgs)

        self.assertIn(0, bp, "messages[-3] (index 0) should have breakpoint")
        self.assertIn(len(msgs) - 1, bp, "messages[-1] should have breakpoint")

    def test_parallel_tool_calls_add_two_messages(self):
        """Parallel tool calls (N tool_use + N tool_result) still produce 2 messages."""
        base = [Message(role="user", content="Analyze data")]

        assistant = Message(
            role="assistant",
            parts=[
                MessagePart(type="text", content="Running queries."),
                MessagePart(
                    type="function_call",
                    function_call={"name": "run_query", "arguments": {"sql": "Q1"}},
                    function_call_id="p1",
                ),
                MessagePart(
                    type="function_call",
                    function_call={"name": "run_query", "arguments": {"sql": "Q2"}},
                    function_call_id="p2",
                ),
            ],
        )
        results = Message(
            role="function",
            parts=[
                MessagePart(type="function_result", content="R1", function_call_id="p1"),
                MessagePart(type="function_result", content="R2", function_call_id="p2"),
            ],
        )

        self.assertEqual(
            len(base + [assistant, results]) - len(base),
            2,
            "Parallel tool calls must add exactly 2 messages",
        )


class _FakeAnthropicMessage:
    """Shared fake for tests that call fetch_async."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }


class TestEmptyTextBlockFiltering(unittest.TestCase):
    """Tests that empty text content blocks are filtered on deserialization and serialization."""

    def test_from_anthropic_dict_skips_empty_text(self):
        """Empty text blocks in API responses should be skipped during deserialization."""
        msg = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "text", "text": ""},
                {"type": "text", "text": "hello"},
            ],
        )
        self.assertEqual(len(msg.parts), 1)
        self.assertEqual(msg.parts[0].content, "hello")

    def test_from_anthropic_dict_skips_whitespace_only_text(self):
        """Whitespace-only text blocks should be skipped during deserialization."""
        msg = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "text", "text": "   \n\t  "},
                {"type": "text", "text": "real content"},
            ],
        )
        self.assertEqual(len(msg.parts), 1)
        self.assertEqual(msg.parts[0].content, "real content")

    def test_from_anthropic_dict_preserves_tool_use_with_empty_text(self):
        """A message with empty text + tool_use should preserve the tool_use part."""
        msg = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "text", "text": ""},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "query",
                    "input": {"sql": "SELECT 1"},
                },
            ],
        )
        self.assertEqual(len(msg.parts), 1)
        self.assertEqual(msg.parts[0].type, "function_call")

    def test_message_to_anthropic_dict_skips_empty_text(self):
        """Empty text parts should be skipped when serializing to API format."""
        from agentlys.providers.anthropic import message_to_anthropic_dict

        msg = Message(role="assistant", parts=[
            MessagePart(type="text", content=""),
            MessagePart(type="text", content="hello"),
        ])
        result = message_to_anthropic_dict(msg)
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "hello")

    def test_message_to_anthropic_dict_skips_whitespace_only_text(self):
        """Whitespace-only text parts should be skipped when serializing."""
        from agentlys.providers.anthropic import message_to_anthropic_dict

        msg = Message(role="assistant", parts=[
            MessagePart(type="text", content="  \n "),
            MessagePart(type="text", content="valid"),
        ])
        result = message_to_anthropic_dict(msg)
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "valid")

    def test_round_trip_filters_empty_text(self):
        """API response with empty text -> deserialize -> serialize -> no empty text."""
        from agentlys.providers.anthropic import message_to_anthropic_dict

        # Simulate API response with empty text block alongside real content
        api_response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "query",
                    "input": {"sql": "SELECT 1"},
                },
            ],
        }

        # Deserialize
        msg = Message.from_anthropic_dict(**api_response)

        # Serialize back
        result = message_to_anthropic_dict(msg)

        # No empty text blocks in output
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "I'll help you with that.")

        # Tool use is preserved
        tool_blocks = [b for b in result["content"] if b.get("type") == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)


if __name__ == "__main__":
    unittest.main()
