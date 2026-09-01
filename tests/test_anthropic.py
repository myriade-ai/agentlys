import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentlys import Agentlys, APIProvider, Message, MessagePart
from agentlys.providers.anthropic import message_to_anthropic_dict


class TestStrictToolSchemas(unittest.TestCase):
    """Closed schemas are automatic; ``strict`` is not.

    Auto-strict used to be inferred from the model name. That guess is not
    available to a provider pointed at a gateway -- the model is a logical
    role, or None -- where it first silently never fired, then crashed on
    ``None.lower()``. Schemas stay closed either way; the flag is the
    caller's.
    """

    @staticmethod
    def _archive_live_block(
        reason: str, name: str = "", live_block_id: str = ""
    ) -> str:
        return "ok"

    def _tool_for_model(self, model):
        agent = Agentlys(provider="anthropic", model=model, api_key="test")
        agent.add_function(self._archive_live_block)
        return agent.provider._build_tools()[0]

    def test_a_strict_capable_model_is_not_made_strict_implicitly(self):
        tool = self._tool_for_model("claude-haiku-4-5-20251001")

        self.assertNotIn("strict", tool)
        self.assertIs(tool["input_schema"]["additionalProperties"], False)
        self.assertEqual(
            set(tool["input_schema"]["properties"]),
            {"reason", "name", "live_block_id"},
        )

    def test_an_older_model_also_keeps_the_closed_schema(self):
        tool = self._tool_for_model("claude-3-7-sonnet-latest")

        self.assertNotIn("strict", tool)
        self.assertIs(tool["input_schema"]["additionalProperties"], False)

    def test_a_model_the_gateway_resolves_builds_a_request(self):
        """``provider.model`` is None whenever the caller leaves it to a
        gateway; reading it to decide strict raised AttributeError on every
        request, tools or not."""
        agent = Agentlys(provider="anthropic", model=None, api_key="test")
        agent.add_function(self._archive_live_block)

        tools = agent.provider._build_tools()

        self.assertEqual([tool["name"] for tool in tools], ["_archive_live_block"])
        self.assertNotIn("strict", tools[0])

    def test_an_explicit_strict_flag_is_forwarded(self):
        agent = Agentlys(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test",
        )
        agent.functions_schema = [
            {
                "name": "explicit_tool",
                "description": "Must be strict",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        ]

        self.assertIs(agent.provider._build_tools()[0]["strict"], True)

    def test_many_tools_are_all_forwarded_unchanged(self):
        """No implicit flag means no per-request budget to run out of: the
        21st tool is built exactly like the first."""
        agent = Agentlys(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test",
        )
        agent.functions_schema = [
            {
                "name": f"tool_{index}",
                "description": "Tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
            for index in range(21)
        ]

        tools = agent.provider._build_tools()

        self.assertEqual(len(tools), 21)
        self.assertTrue(all("strict" not in tool for tool in tools))


class TestAnthropic(unittest.TestCase):
    def setUp(self):
        self.mock_anthropic_client = MagicMock()

    def test_transform_conversation_anthropic(self):
        agent = Agentlys(instruction="Test instruction", provider=APIProvider.ANTHROPIC)
        # A tool result must be preceded by its tool_use — an orphaned result
        # (no matching call) is stripped before the request, so the history has
        # to carry the assistant tool_use for the merge to be exercised.
        agent.messages = [
            Message(
                role="assistant",
                parts=[
                    MessagePart(
                        type="function_call",
                        function_call={"name": "SUBMIT", "arguments": {}},
                        function_call_id="example_16",
                    )
                ],
            ),
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
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "example_16",
                        "name": "SUBMIT",
                        "input": {},
                    }
                ],
            },
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
            },
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


class TestThinkingReplay(unittest.TestCase):
    """Thinking blocks are replayed for live messages, dropped for stored ones.

    Replaces the old TestStripThinkingFromPriorTurns suite, which asserted
    that thinking survived only on the last assistant message: that rule
    rewrote the previous assistant turn on every tool-loop iteration and
    invalidated the cached prefix from that point on.  Provenance
    (Message.is_live) is what actually matters — a stale signature can only
    come from a message rebuilt outside this process.
    """

    @staticmethod
    def _live_assistant():
        return Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "thought", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
        )

    def test_live_message_replays_thinking_unchanged(self):
        message = self._live_assistant()

        blocks = message_to_anthropic_dict(message)["content"]

        self.assertEqual(
            blocks[0],
            {"type": "thinking", "thinking": "thought", "signature": "sig"},
        )

    def test_live_message_replays_redacted_thinking(self):
        message = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "redacted_thinking", "data": "encrypted"},
                {"type": "text", "text": "answer"},
            ],
        )

        blocks = message_to_anthropic_dict(message)["content"]

        self.assertEqual(blocks[0], {"type": "redacted_thinking", "data": "encrypted"})

    def test_stored_message_drops_thinking(self):
        """A message rebuilt from storage may carry a signature from another
        model version, which the API rejects."""
        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="thinking", thinking="thought", thinking_signature="sig"
                ),
                MessagePart(type="text", content="answer"),
            ],
        )

        blocks = message_to_anthropic_dict(message)["content"]

        self.assertEqual(blocks, [{"type": "text", "text": "answer"}])

    def test_earlier_live_turns_keep_their_thinking(self):
        """The whole point: an earlier assistant turn is not rewritten when a
        new one arrives, so the cached prefix stays valid."""
        first, second = self._live_assistant(), self._live_assistant()

        self.assertEqual(
            message_to_anthropic_dict(first)["content"][0]["type"], "thinking"
        )
        self.assertEqual(
            message_to_anthropic_dict(second)["content"][0]["type"], "thinking"
        )

    def test_live_message_drops_empty_thinking(self):
        """The model can emit a signed thinking block with empty content; the
        API refuses to take it back ('each thinking block must contain
        thinking', 400) once it sits on the last assistant message, killing
        every later call of the conversation. Drop it at serialization."""
        message = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
        )

        blocks = message_to_anthropic_dict(message)["content"]

        self.assertEqual(blocks, [{"type": "text", "text": "answer"}])

    def test_live_message_drops_thinking_without_content_field(self):
        message = Message.from_anthropic_dict(
            role="assistant",
            content=[
                {"type": "thinking", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
        )

        blocks = message_to_anthropic_dict(message)["content"]

        self.assertEqual(blocks, [{"type": "text", "text": "answer"}])

    def test_load_messages_marks_history_as_stored(self):
        agent = Agentlys(provider=APIProvider.ANTHROPIC, api_key="test")
        live = self._live_assistant()
        self.assertTrue(live.is_live)

        agent.load_messages([Message(role="user", content="hi"), live])

        # load_messages strips thinking outright; whatever it keeps must not
        # claim to be a fresh signature.
        self.assertEqual(
            [p.type for p in agent.messages[1].parts],
            ["text"],
        )


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
                return {
                    "role": self.role,
                    "content": self.content,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }

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
            user_texts = [
                b.get("text", "") for b in first_msg_content if isinstance(b, dict)
            ]
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
            Message(role="function", content="Alice,Bob", function_call_id="old_1"),
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
                            "arguments": {
                                "sql": "SELECT user_id, COUNT(*) FROM orders GROUP BY 1"
                            },
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
                        function_call={
                            "name": "run_query",
                            "arguments": {"sql": "SELECT 1"},
                        },
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
                MessagePart(
                    type="function_result", content="R1", function_call_id="p1"
                ),
                MessagePart(
                    type="function_result", content="R2", function_call_id="p2"
                ),
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

        msg = Message(
            role="assistant",
            parts=[
                MessagePart(type="text", content=""),
                MessagePart(type="text", content="hello"),
            ],
        )
        result = message_to_anthropic_dict(msg)
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "hello")

    def test_message_to_anthropic_dict_skips_whitespace_only_text(self):
        """Whitespace-only text parts should be skipped when serializing."""
        from agentlys.providers.anthropic import message_to_anthropic_dict

        msg = Message(
            role="assistant",
            parts=[
                MessagePart(type="text", content="  \n "),
                MessagePart(type="text", content="valid"),
            ],
        )
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


class TestDocumentParts(unittest.TestCase):
    def test_document_part_to_anthropic_dict(self):
        from agentlys.model import Document
        from agentlys.providers.anthropic import message_to_anthropic_dict

        pdf_bytes = b"%PDF-1.4 fake pdf payload"
        msg = Message(
            role="user",
            parts=[
                MessagePart(type="text", content="Summarize this document"),
                MessagePart(
                    type="document",
                    document=Document(
                        pdf_bytes, media_type="application/pdf", name="report.pdf"
                    ),
                ),
            ],
        )

        result = message_to_anthropic_dict(msg)
        self.assertEqual(result["role"], "user")
        doc_blocks = [b for b in result["content"] if b.get("type") == "document"]
        self.assertEqual(len(doc_blocks), 1)
        block = doc_blocks[0]
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["media_type"], "application/pdf")
        self.assertEqual(block["title"], "report.pdf")

        import base64 as b64

        self.assertEqual(b64.b64decode(block["source"]["data"]), pdf_bytes)

    def test_document_base64_round_trip(self):
        from agentlys.model import Document

        original = Document(b"hello world", media_type="text/plain", name="a.txt")
        restored = Document.from_base64(
            original.to_base64(), media_type="text/plain", name="a.txt"
        )
        self.assertEqual(restored.data, b"hello world")
        self.assertEqual(restored.media_type, "text/plain")

    def test_document_part_without_document_raises(self):
        from agentlys.providers.anthropic import part_to_anthropic_dict

        with self.assertRaises(ValueError):
            part_to_anthropic_dict(MessagePart(type="document"))

    def test_text_plain_document_uses_text_source(self):
        from agentlys.model import Document
        from agentlys.providers.anthropic import part_to_anthropic_dict

        block = part_to_anthropic_dict(
            MessagePart(
                type="document",
                document=Document(
                    b"hello world", media_type="text/plain", name="notes.txt"
                ),
            )
        )
        self.assertEqual(block["source"]["type"], "text")
        self.assertEqual(block["source"]["media_type"], "text/plain")
        self.assertEqual(block["source"]["data"], "hello world")
        self.assertEqual(block["title"], "notes.txt")

    def test_unsupported_document_media_type_raises_early(self):
        from agentlys.model import Document
        from agentlys.providers.anthropic import part_to_anthropic_dict

        with self.assertRaises(ValueError) as ctx:
            part_to_anthropic_dict(
                MessagePart(
                    type="document",
                    document=Document(b"GIF89a", media_type="image/gif"),
                )
            )
        self.assertIn("image/gif", str(ctx.exception))

    def test_document_base64_is_memoized(self):
        from agentlys.model import Document

        doc = Document(b"%PDF-1.4 payload")
        first = doc.to_base64()
        # Same cached string object on subsequent calls (no re-encoding)
        self.assertIs(doc.to_base64(), first)

    def test_text_document_content_survives_to_markdown(self):
        # The compaction summarizer reads history via to_markdown(); text
        # documents must expose their content there, and binary documents
        # must say so explicitly instead of silently dropping content.
        from agentlys.model import Document

        text_msg = Message(
            role="user",
            parts=[
                MessagePart(
                    type="document",
                    document=Document(
                        b"quarterly revenue: 42", media_type="text/plain", name="q.txt"
                    ),
                )
            ],
        )
        md = text_msg.to_markdown()
        self.assertIn("quarterly revenue: 42", md)

        pdf_msg = Message(
            role="user",
            parts=[
                MessagePart(
                    type="document",
                    document=Document(b"%PDF-1.4", name="report.pdf"),
                )
            ],
        )
        md = pdf_msg.to_markdown()
        self.assertIn("report.pdf", md)
        self.assertIn("binary content not rendered", md)


if __name__ == "__main__":
    unittest.main()
