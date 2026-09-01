"""Prompt-cache stability: what the provider sends must not drift mid-turn.

Anthropic's cache is a prefix match, so any byte that changes between two
iterations of the same turn invalidates everything after it.  These tests pin
the four things that used to move (system tool states, user_context position,
thinking blocks, breakpoint TTLs) plus the effort and debug knobs.
"""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from agentlys import Agentlys, APIProvider, Message
from agentlys.model import MessagePart
from agentlys.utils import get_event_loop_or_create


def _run(coro):
    """Drive *coro* to completion, leaving the thread's event loop installed.

    asyncio.run() unsets the current event loop on its way out, so a test that
    later reaches for the ambient one — test_anthropic and test_parse_template
    both call bare asyncio.get_event_loop() — would fail with "no current event
    loop" purely because of the order the files happen to run in.
    """
    return get_event_loop_or_create().run_until_complete(coro)


class LiveTool:
    """A tool whose state changes while the turn is running."""

    def __init__(self):
        self.calls = 0

    def __llm__(self):
        return f"LiveTool(calls={self.calls})"

    def ping(self) -> str:
        """Ping the tool.

        Returns: the string "pong"
        """
        self.calls += 1
        return "pong"


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def to_dict(self):
        return {
            "role": "assistant",
            "content": self._content,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 100,
            },
        }


def _agent(**kwargs):
    agent = Agentlys(
        instruction="You are a test agent.",
        provider=APIProvider.ANTHROPIC,
        model="claude-sonnet-4-5",
        api_key="test",
        **kwargs,
    )
    agent.messages = [Message(role="user", content="hello")]
    return agent


def _prepare(agent, **kwargs):
    return agent.provider._prepare_request_params(**kwargs)


def _content_only(payload):
    """Drop cache_control markers.

    Breakpoints are meant to move forward from one iteration to the next —
    that is how the previous prefix gets read instead of rewritten. What must
    not change is the content they sit on.
    """
    if isinstance(payload, dict):
        return {k: _content_only(v) for k, v in payload.items() if k != "cache_control"}
    if isinstance(payload, list):
        return [_content_only(item) for item in payload]
    return payload


class TestIntraTurnStability(unittest.TestCase):
    """Two consecutive requests of one turn must share identical bytes."""

    def _run_turn(self):
        agent = Agentlys(
            instruction="You are a test agent.",
            context="Static context.",
            user_context="Untrusted project description.",
            provider=APIProvider.ANTHROPIC,
            model="claude-sonnet-4-5",
            api_key="test",
        )
        tool = LiveTool()
        tool_id = agent.add_tool(tool)
        function_name = f"LiveTool-{tool_id}__ping"

        # Two tool iterations: three requests, so the second request's own
        # assistant turn becomes a *prior* turn in the third one.
        responses = [
            FakeResponse(
                [
                    {"type": "thinking", "thinking": "ping once", "signature": "s1"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": function_name,
                        "input": {},
                    },
                ]
            ),
            FakeResponse(
                [
                    {"type": "thinking", "thinking": "ping twice", "signature": "s2"},
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": function_name,
                        "input": {},
                    },
                ]
            ),
            FakeResponse([{"type": "text", "text": "done"}]),
        ]
        create = AsyncMock(side_effect=responses)

        async def drive():
            with patch.object(agent.provider.client.messages, "create", create):
                async for _ in agent.run_conversation_async("hello"):
                    pass

        _run(drive())
        self.assertEqual(create.await_count, 3)
        self.assertEqual(tool.calls, 2, "the tool must have run between the calls")
        return [call.kwargs for call in create.await_args_list]

    def test_system_and_tools_are_byte_identical(self):
        requests = self._run_turn()

        # The tool's __llm__ moved from calls=0 to calls=2 while the turn ran;
        # the system prompt must still show the snapshot taken for this turn.
        self.assertIn("LiveTool(calls=0)", json.dumps(requests[0]["system"]))
        for request in requests[1:]:
            self.assertEqual(
                json.dumps(request["system"]), json.dumps(requests[0]["system"])
            )
            self.assertEqual(
                json.dumps(request["tools"]), json.dumps(requests[0]["tools"])
            )

    def test_message_prefix_is_byte_identical(self):
        requests = self._run_turn()

        # Each iteration appends exactly one assistant + one tool_result
        # message; everything it inherits must be untouched.
        for previous, current in zip(requests, requests[1:]):
            shared = len(current["messages"]) - 2
            self.assertEqual(shared, len(previous["messages"]))
            self.assertEqual(
                json.dumps(_content_only(current["messages"][:shared])),
                json.dumps(_content_only(previous["messages"][:shared])),
            )

    def test_user_context_stays_on_the_first_user_message(self):
        for request in self._run_turn():
            blocks = request["messages"][0]["content"]
            self.assertEqual(blocks[0]["text"], "Untrusted project description.")
            # Exactly once, and never on a tool_result message.
            dumped = json.dumps(request["messages"])
            self.assertEqual(dumped.count("Untrusted project description."), 1)

    def test_thinking_is_replayed_unchanged_in_the_tool_loop(self):
        last = self._run_turn()[-1]

        # Both assistant turns of the loop keep their thinking block: the API
        # requires it, and rewriting the earlier one is what used to break the
        # cached prefix on every iteration.
        self.assertEqual(
            last["messages"][1]["content"][0],
            {"type": "thinking", "thinking": "ping once", "signature": "s1"},
        )
        self.assertEqual(
            last["messages"][3]["content"][0],
            {"type": "thinking", "thinking": "ping twice", "signature": "s2"},
        )


def _storage_roundtrip(messages):
    """Rebuild the history the way a consumer reloading it from a DB would.

    Same shape as agentlys' own model, but freshly built objects: no is_live,
    exactly what a caller gets back from its own tables.
    """
    rebuilt = []
    for message in messages:
        parts = [
            MessagePart(
                type=part.type,
                content=part.content,
                function_call=part.function_call,
                function_call_id=part.function_call_id,
                thinking=part.thinking,
                thinking_signature=part.thinking_signature,
                is_redacted=part.is_redacted,
            )
            for part in message.parts
        ]
        rebuilt.append(Message(role=message.role, name=message.name, parts=parts))
    return rebuilt


class TestToolCallSerialization(unittest.TestCase):
    def test_tool_use_input_keys_are_in_canonical_order(self):
        from agentlys.providers.anthropic import part_to_anthropic_dict

        part = MessagePart(
            type="function_call",
            function_call={
                "name": "sql_query",
                "arguments": {
                    "query": "select 1",
                    "database_id": 3,
                    "a": [{"z": 1, "b": 2}],
                },
                "id": "t1",
            },
            function_call_id="t1",
        )

        block = part_to_anthropic_dict(part)

        # A store that reorders object keys (Postgres jsonb) must not change
        # the bytes of a reloaded tool call.
        self.assertEqual(list(block["input"]), ["a", "database_id", "query"])
        self.assertEqual(list(block["input"]["a"][0]), ["b", "z"])


class TestCrossTurnStability(unittest.TestCase):
    """A reloaded conversation must serialize exactly as it was sent."""

    @staticmethod
    def _ping() -> str:
        """Ping.

        Returns: the string "pong"
        """
        return "pong"

    def _agent_with_tool(self):
        agent = Agentlys(
            instruction="You are a test agent.",
            context="Static context.",
            provider=APIProvider.ANTHROPIC,
            model="claude-sonnet-4-5",
            api_key="test",
        )
        agent.add_function(self._ping)
        return agent

    def _turn_one(self):
        """Run a full turn and return (last request, agent messages)."""
        agent = self._agent_with_tool()
        responses = [
            FakeResponse(
                [
                    {"type": "thinking", "thinking": "let me ping", "signature": "s1"},
                    {"type": "tool_use", "id": "t1", "name": "_ping", "input": {}},
                ]
            ),
            FakeResponse(
                [
                    {"type": "thinking", "thinking": "got pong", "signature": "s2"},
                    {"type": "text", "text": "it answered pong"},
                ]
            ),
        ]
        create = AsyncMock(side_effect=responses)

        async def drive():
            with patch.object(agent.provider.client.messages, "create", create):
                async for _ in agent.run_conversation_async("first question"):
                    pass

        _run(drive())
        self.assertEqual(create.await_count, 2)
        return create.await_args_list[-1].kwargs, agent.messages

    def _turn_two(self, history, keep_thinking):
        """Reload `history` from storage, ask again, return the request."""
        agent = self._agent_with_tool()
        agent.load_messages(_storage_roundtrip(history), keep_thinking=keep_thinking)
        create = AsyncMock(side_effect=[FakeResponse([{"type": "text", "text": "ok"}])])

        async def drive():
            with patch.object(agent.provider.client.messages, "create", create):
                async for _ in agent.run_conversation_async("second question"):
                    pass

        _run(drive())
        return create.await_args_list[0].kwargs

    def test_reloaded_turn_serializes_byte_identically(self):
        first, history = self._turn_one()
        second = self._turn_two(history, keep_thinking=True)

        # Turn 1 sent 3 messages (question, assistant+tool_use, tool_result);
        # its last request also carried the trailing assistant message, which
        # only exists in the history. Compare what both requests share.
        shared = len(first["messages"])
        self.assertEqual(len(second["messages"]), shared + 2)
        self.assertEqual(
            json.dumps(_content_only(second["messages"][:shared])),
            json.dumps(_content_only(first["messages"][:shared])),
        )

        # And the thinking blocks are the ones the API produced, in place.
        self.assertEqual(
            second["messages"][1]["content"][0],
            {"type": "thinking", "thinking": "let me ping", "signature": "s1"},
        )

    def test_without_the_opt_in_the_prefix_breaks_at_the_first_answer(self):
        first, history = self._turn_one()
        second = self._turn_two(history, keep_thinking=False)

        # The regression this guards: the assistant message loses its thinking
        # block, so every byte from there on is rewritten on the new question.
        self.assertNotEqual(
            json.dumps(_content_only(second["messages"][1])),
            json.dumps(_content_only(first["messages"][1])),
        )

    def test_unsigned_thinking_is_dropped_even_when_kept(self):
        agent = self._agent_with_tool()
        agent.load_messages(
            [
                Message(role="user", content="q"),
                Message(
                    role="assistant",
                    parts=[
                        MessagePart(type="thinking", thinking="legacy row"),
                        MessagePart(type="text", content="answer"),
                    ],
                ),
            ],
            keep_thinking=True,
        )

        messages, _, _ = _prepare(agent)

        # Rows persisted before signatures were stored would be rejected.
        self.assertEqual(
            _content_only(messages[1]["content"]),
            [{"type": "text", "text": "answer"}],
        )
        self.assertFalse(agent.messages[1].is_live)

    def test_redacted_thinking_survives_the_reload(self):
        agent = self._agent_with_tool()
        agent.load_messages(
            [
                Message(role="user", content="q"),
                Message(
                    role="assistant",
                    parts=[
                        MessagePart(
                            type="thinking",
                            thinking=None,
                            thinking_signature="encrypted-blob",
                            is_redacted=True,
                        ),
                        MessagePart(type="text", content="answer"),
                    ],
                ),
            ],
            keep_thinking=True,
        )

        messages, _, _ = _prepare(agent)

        self.assertEqual(
            messages[1]["content"][0],
            {"type": "redacted_thinking", "data": "encrypted-blob"},
        )


class TestUserContextPosition(unittest.TestCase):
    def test_stays_on_the_first_user_message_across_turns(self):
        agent = _agent(user_context="Project description.")
        agent.messages = [
            Message(role="user", content="first question"),
            Message(role="assistant", content="first answer"),
            Message(role="user", content="second question"),
        ]

        messages, _, _ = _prepare(agent)

        # Anchored to the first human message, so the whole prefix built
        # during turn 1 survives turn 2 instead of being rewritten.
        self.assertEqual(messages[0]["content"][0]["text"], "Project description.")
        self.assertEqual(json.dumps(messages).count("Project description."), 1)
        self.assertEqual(messages[2]["content"][0]["text"], "second question")

    def test_examples_are_never_patched(self):
        agent = _agent(user_context="Project description.")
        agent.examples = [Message(role="user", content="example question")]
        agent.messages = [Message(role="user", content="real question")]

        messages, _, _ = _prepare(agent)

        # Consecutive user messages are merged into one, so assert on the
        # block order: the example comes first, untouched.
        texts = [block["text"] for block in messages[0]["content"]]
        self.assertEqual(
            texts, ["example question", "Project description.", "real question"]
        )


class TestToolStatesSnapshot(unittest.TestCase):
    def test_snapshot_is_taken_again_on_a_new_user_message(self):
        agent = _agent()
        tool = LiveTool()
        agent.add_tool(tool)

        agent._capture_tools_states(Message(role="user", content="q1"))
        self.assertIn("calls=0", agent.initial_tools_states)

        tool.calls = 3
        # A tool result must not move the snapshot...
        agent._capture_tools_states(Message(role="function", content="pong"))
        self.assertIn("calls=0", agent.initial_tools_states)

        # ...but the next human turn must see the current state.
        agent._capture_tools_states(Message(role="user", content="q2"))
        self.assertIn("calls=3", agent.initial_tools_states)

    def test_refresh_and_reset_clear_the_snapshot(self):
        agent = _agent()
        tool = LiveTool()
        agent.add_tool(tool)
        agent._capture_tools_states(Message(role="user", content="q"))

        tool.calls = 7
        agent.refresh_tools_states()

        self.assertIn("calls=7", agent.initial_tools_states)


class TestCacheTTL(unittest.TestCase):
    def _breakpoints(self, agent, **kwargs):
        messages, tools, kwargs = _prepare(agent, **kwargs)
        system = kwargs["system"]
        return {
            "system": system[-1]["cache_control"],
            "tools": [t["cache_control"] for t in tools if "cache_control" in t],
            "messages": [
                block["cache_control"]
                for message in messages
                for block in message["content"]
                if isinstance(block, dict) and "cache_control" in block
            ],
            "kwargs": kwargs,
        }

    def test_default_is_unchanged(self):
        agent = _agent()
        agent.add_tool(LiveTool())

        found = self._breakpoints(agent)

        self.assertEqual(found["system"], {"type": "ephemeral"})
        self.assertEqual(found["tools"], [{"type": "ephemeral"}])
        self.assertEqual(found["messages"], [{"type": "ephemeral"}])
        self.assertNotIn("extra_headers", found["kwargs"])

    def test_one_hour_ttl_on_every_breakpoint(self):
        agent = _agent(cache_ttl="1h")
        agent.add_tool(LiveTool())

        found = self._breakpoints(agent)

        self.assertEqual(found["system"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(found["tools"], [{"type": "ephemeral", "ttl": "1h"}])
        self.assertEqual(found["messages"], [{"type": "ephemeral", "ttl": "1h"}])

    def test_messages_ttl_is_configured_separately(self):
        agent = _agent(cache_ttl="1h", cache_ttl_messages="5m")

        found = self._breakpoints(agent)

        self.assertEqual(found["system"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(found["messages"], [{"type": "ephemeral"}])

    def test_env_vars_configure_the_ttls(self):
        with patch.dict(
            os.environ,
            {"AGENTLYS_CACHE_TTL": "1h", "AGENTLYS_CACHE_TTL_MESSAGES": "5m"},
        ):
            agent = _agent()

        found = self._breakpoints(agent)

        self.assertEqual(found["system"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(found["messages"], [{"type": "ephemeral"}])

    def test_longer_message_ttl_than_system_ttl_is_clamped(self):
        # The API requires longer-lived entries first, and tools + system are
        # rendered before messages.
        with self.assertLogs("agentlys.providers.anthropic", level="WARNING"):
            agent = _agent(cache_ttl_messages="1h")

        found = self._breakpoints(agent)

        self.assertEqual(found["messages"], [{"type": "ephemeral"}])

    def test_invalid_ttl_is_rejected(self):
        with self.assertRaises(ValueError):
            _agent(cache_ttl="10m")

    def test_extended_ttl_beta_is_advertised(self):
        agent = _agent(cache_ttl="1h")

        _, _, kwargs = _prepare(agent)

        self.assertEqual(
            kwargs["extra_headers"]["anthropic-beta"],
            "extended-cache-ttl-2025-04-11",
        )

    def test_extended_ttl_beta_preserves_existing_headers(self):
        agent = _agent(cache_ttl="1h")

        _, _, kwargs = _prepare(
            agent,
            extra_headers={"X-Session": "abc", "anthropic-beta": "other-beta"},
        )

        self.assertEqual(kwargs["extra_headers"]["X-Session"], "abc")
        self.assertEqual(
            kwargs["extra_headers"]["anthropic-beta"],
            "other-beta,extended-cache-ttl-2025-04-11",
        )

    def test_extended_ttl_beta_can_be_disabled(self):
        agent = _agent(cache_ttl="1h")

        with patch.dict(os.environ, {"AGENTLYS_CACHE_TTL_BETA": "0"}):
            _, _, kwargs = _prepare(agent)

        self.assertNotIn("extra_headers", kwargs)


class TestEffort(unittest.TestCase):
    def test_not_sent_by_default(self):
        _, _, kwargs = _prepare(_agent())

        self.assertNotIn("extra_body", kwargs)

    def test_constructor_effort(self):
        _, _, kwargs = _prepare(_agent(effort="medium"))

        self.assertEqual(kwargs["extra_body"], {"output_config": {"effort": "medium"}})

    def test_env_effort(self):
        with patch.dict(os.environ, {"AGENTLYS_EFFORT": "low"}):
            agent = _agent()

        _, _, kwargs = _prepare(agent)

        self.assertEqual(kwargs["extra_body"], {"output_config": {"effort": "low"}})

    def test_per_call_override_wins(self):
        _, _, kwargs = _prepare(_agent(effort="low"), effort="max")

        self.assertEqual(kwargs["extra_body"], {"output_config": {"effort": "max"}})

    def test_invalid_effort_is_rejected(self):
        with self.assertRaises(ValueError):
            _agent(effort="turbo")


class TestCacheDebug(unittest.TestCase):
    def test_hashes_and_usage_are_logged(self):
        with patch.dict(os.environ, {"AGENTLYS_CACHE_DEBUG": "1"}):
            agent = _agent()
            agent.add_tool(LiveTool())
            create = AsyncMock(
                return_value=FakeResponse([{"type": "text", "text": "hi"}])
            )

            with self.assertLogs("agentlys.providers.anthropic", level="INFO") as logs:
                with patch.object(agent.provider.client.messages, "create", create):
                    _run(agent.provider.fetch_async())

        request_log = next(m for m in logs.output if "cache request" in m)
        usage_log = next(m for m in logs.output if "cache usage" in m)
        self.assertIn("system=", request_log)
        self.assertIn("tools=", request_log)
        self.assertIn("messages=", request_log)
        self.assertIn("cache_read_input_tokens", usage_log)

    def test_silent_by_default(self):
        agent = _agent()
        _prepare(agent)  # must not raise, must not log


class TestIsLiveSurvivesRebuilds(unittest.TestCase):
    """Every place that rebuilds a Message must carry ``is_live`` over.

    ``Message.__init__`` does not take it, so a rebuild silently demotes the
    message: its thinking blocks stop being replayed and the assistant turn
    they belong to is rewritten, invalidating the cached prefix from there on.
    """

    def test_dropping_an_orphaned_result_keeps_thinking_replayable(self):
        from agentlys.providers.anthropic import message_to_anthropic_dict
        from agentlys.providers.utils import drop_orphaned_function_results

        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="thinking", thinking="reasoned", thinking_signature="s1"
                ),
                MessagePart(type="text", content="answer"),
                MessagePart(
                    type="function_result", content="stale", function_call_id="gone"
                ),
            ],
        )
        message.is_live = True

        kept = drop_orphaned_function_results([message])

        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].is_live)
        self.assertEqual(
            message_to_anthropic_dict(kept[0])["content"][0],
            {"type": "thinking", "thinking": "reasoned", "signature": "s1"},
        )

    def test_user_context_patch_keeps_the_flag(self):
        agent = _agent(user_context="Project description.")
        original = Message(role="user", content="hello")
        original.is_live = True
        agent.messages = [original]

        prepared = agent.provider.prepare_messages(transform_function=lambda m: m)

        self.assertEqual(prepared[0].parts[0].content, "Project description.")
        self.assertTrue(prepared[0].is_live)


def _live_assistant(thinking, signature, text=None):
    parts = [
        MessagePart(type="thinking", thinking=thinking, thinking_signature=signature)
    ]
    if text is not None:
        parts.append(MessagePart(type="text", content=text))
    message = Message(role="assistant", parts=parts)
    message.is_live = True
    return message


class TestMergedAssistantThinking(unittest.TestCase):
    def test_two_thinking_turns_merge_into_a_valid_content_list(self):
        """Merging two assistant turns must keep each thinking block in place.

        The API accepts several thinking blocks inside one assistant turn
        (that is what interleaved thinking produces), and dropping a regular
        thinking block is what triggers ordering/signature 400s — so both are
        replayed, each still ahead of the text it produced.
        """
        agent = _agent()
        agent.messages = [
            Message(role="user", content="q"),
            _live_assistant("one", "s1", "a"),
            _live_assistant("two", "s2", "b"),
        ]

        messages, _, _ = _prepare(agent)

        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(
            _content_only(messages[1]["content"]),
            [
                {"type": "thinking", "thinking": "one", "signature": "s1"},
                {"type": "text", "text": "a"},
                {"type": "thinking", "thinking": "two", "signature": "s2"},
                {"type": "text", "text": "b"},
            ],
        )


class TestThinkingOnlyAssistantMessage(unittest.TestCase):
    """An assistant turn holding nothing but a thinking block."""

    def test_replayable_one_is_serialized(self):
        agent = _agent()
        agent.messages = [
            Message(role="user", content="q"),
            _live_assistant("alone", "s1"),
        ]

        messages, _, _ = _prepare(agent)

        self.assertEqual(
            _content_only(messages[1]["content"]),
            [{"type": "thinking", "thinking": "alone", "signature": "s1"}],
        )

    def test_non_replayable_one_is_dropped_not_sent_empty(self):
        agent = _agent()
        stale = Message(
            role="assistant",
            parts=[
                MessagePart(type="thinking", thinking="alone", thinking_signature="s1")
            ],
        )
        agent.messages = [
            Message(role="user", content="q1"),
            stale,
            Message(role="user", content="q2"),
        ]

        messages, _, _ = _prepare(agent)

        # Its only block is stripped, so the turn has nothing left to say:
        # it is dropped, never sent as an assistant turn with content=[].
        for message in messages:
            self.assertTrue(message["content"], message)
        self.assertNotIn("assistant", [m["role"] for m in messages])



class TestPinnedToolsStates(unittest.TestCase):
    """pin_tools_states(): a caller-provided snapshot survives new turns.

    Without a pin, every new user message recaptures the live __llm__()
    values, so any drift (a documented table, a moved counter) rewrites the
    system prompt and invalidates the whole cached prefix on the next turn.
    A caller that persists the block across turns pins it back instead.
    """

    def _agent_with_live_tool(self):
        agent = Agentlys(
            instruction="You are a test agent.",
            provider=APIProvider.ANTHROPIC,
            model="claude-sonnet-4-5",
            api_key="test",
        )
        tool = LiveTool()
        agent.add_tool(tool)
        return agent, tool

    def _drive_turn(self, agent, question):
        create = AsyncMock(return_value=FakeResponse([{"type": "text", "text": "ok"}]))

        async def drive():
            with patch.object(agent.provider.client.messages, "create", create):
                async for _ in agent.run_conversation_async(question):
                    pass

        _run(drive())
        return create.await_args_list[-1].kwargs

    def test_a_new_turn_recaptures_by_default(self):
        agent, tool = self._agent_with_live_tool()
        first = self._drive_turn(agent, "q1")
        self.assertIn("LiveTool(calls=0)", json.dumps(first["system"]))

        tool.calls = 7  # state drifts between the turns
        second = self._drive_turn(agent, "q2")
        self.assertIn("LiveTool(calls=7)", json.dumps(second["system"]))

    def test_pinned_snapshot_survives_new_turns(self):
        agent, tool = self._agent_with_live_tool()
        agent.pin_tools_states(agent.initial_tools_states)

        self._drive_turn(agent, "q1")
        tool.calls = 7
        second = self._drive_turn(agent, "q2")

        dumped = json.dumps(second["system"])
        self.assertIn("LiveTool(calls=0)", dumped)
        self.assertNotIn("LiveTool(calls=7)", dumped)

    def test_pin_uses_the_exact_provided_string(self):
        agent, _ = self._agent_with_live_tool()
        stored = (
            "## Initial Tools States\nLiveTool(calls=3, from storage)\n"
            "--- End of Initial Tools States ---"
        )
        agent.pin_tools_states(stored)
        request = self._drive_turn(agent, "q1")
        self.assertIn("LiveTool(calls=3, from storage)", json.dumps(request["system"]))

    def test_refresh_lifts_the_pin(self):
        agent, tool = self._agent_with_live_tool()
        agent.pin_tools_states(agent.initial_tools_states)

        tool.calls = 7
        agent.refresh_tools_states()
        request = self._drive_turn(agent, "q1")
        self.assertIn("LiveTool(calls=7)", json.dumps(request["system"]))

        # The pin is gone for good: the next turn recaptures again.
        tool.calls = 9
        second = self._drive_turn(agent, "q2")
        self.assertIn("LiveTool(calls=9)", json.dumps(second["system"]))

    def test_reset_lifts_the_pin(self):
        agent, _ = self._agent_with_live_tool()
        agent.pin_tools_states("pinned text")
        agent.reset()
        self.assertFalse(agent._tools_states_pinned)
        self.assertIsNone(agent._initial_tools_states)


if __name__ == "__main__":
    unittest.main()
