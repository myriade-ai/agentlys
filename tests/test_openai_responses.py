"""Tests for the OpenAI Responses API provider (GPT-5 / o-series reasoning).

No network: a fake client records the request and replays canned
``responses.create`` results shaped like the real SDK objects.
"""

from types import SimpleNamespace as NS

import pytest
from agentlys import Agentlys
from agentlys.model import Document, Message, MessagePart
from agentlys.providers.openai_responses import (
    OpenAIResponsesProvider,
    decode_thinking_signature,
    encode_thinking_signature,
    message_to_responses_items,
    output_to_message,
    resolve_effort,
    thinking_to_effort,
    usage_to_dict,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "AGENTLYS_HOST",
        "AGENTLYS_API_KEY",
        "AGENTLYS_PROVIDER",
        "AGENTLYS_EFFORT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")


def _usage(input_tokens=100, cached=0, output_tokens=20, reasoning=0):
    return NS(
        input_tokens=input_tokens,
        input_tokens_details=NS(cached_tokens=cached),
        output_tokens=output_tokens,
        output_tokens_details=NS(reasoning_tokens=reasoning),
        total_tokens=input_tokens + output_tokens,
    )


def _reasoning_item(item_id="rs_1", text="thinking hard", encrypted="ENC"):
    return NS(
        type="reasoning",
        id=item_id,
        summary=[NS(type="summary_text", text=text)] if text else [],
        encrypted_content=encrypted,
    )


def _message_item(text):
    return NS(
        type="message",
        id="msg_1",
        role="assistant",
        content=[NS(type="output_text", text=text)],
    )


def _function_call_item(
    call_id="call_1", name="get_weather", arguments='{"city": "Paris"}'
):
    return NS(
        type="function_call", id="fc_1", call_id=call_id, name=name, arguments=arguments
    )


def _response(output, response_id="resp_1", usage=None):
    return NS(
        id=response_id,
        output=output,
        usage=usage or _usage(),
        output_text="".join(
            c.text
            for i in output
            if getattr(i, "type", None) == "message"
            for c in i.content
        ),
        error=None,
    )


class FakeResponses:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if kwargs.get("stream"):
            return _AsyncIter(result)
        return result


class _AsyncIter:
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


def _agent(results, **kwargs):
    agent = Agentlys(provider="openai_responses", model="gpt-5.4", **kwargs)
    fake = FakeResponses(results)
    agent.provider.client = NS(responses=fake)
    return agent, fake


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_provider_selected_by_name(self):
        agent = Agentlys(provider="openai_responses", model="gpt-5.4")
        assert isinstance(agent.provider, OpenAIResponsesProvider)
        assert str(agent.provider.client.base_url) == "https://api.openai.com/v1/"

    def test_default_model(self):
        agent = Agentlys(provider="openai_responses")
        assert agent.provider.model == "gpt-5.4"

    def test_anthropic_style_kwargs_are_accepted(self):
        # The same Agentlys(...) call must work whichever provider is behind it.
        agent = Agentlys(
            provider="openai_responses",
            model="gpt-5.4",
            effort="max",
            cache_ttl="1h",
            cache_ttl_messages="1h",
            thinking={"type": "adaptive"},
        )
        assert agent.provider.effort == "xhigh"

    def test_invalid_effort(self):
        with pytest.raises(ValueError):
            Agentlys(provider="openai_responses", model="gpt-5.4", effort="turbo")

    def test_custom_base_url_and_headers(self):
        agent = Agentlys(provider="openai_responses", model="gpt-5.4")
        provider = OpenAIResponsesProvider(
            agent,
            model="gpt-5.4",
            base_url="http://proxy.local/ai/v1",
            api_key="session-token",
            default_headers={"X-Purpose": "chat"},
        )
        assert str(provider.client.base_url) == "http://proxy.local/ai/v1/"
        assert provider.client.api_key == "session-token"
        assert provider.client.default_headers["X-Purpose"] == "chat"


# ---------------------------------------------------------------------------
# Effort / thinking mapping
# ---------------------------------------------------------------------------


class TestReasoningMapping:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, None),
            ("low", "low"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "xhigh"),
            ("none", "none"),
        ],
    )
    def test_resolve_effort(self, value, expected):
        assert resolve_effort(value) == expected

    @pytest.mark.parametrize(
        "thinking, expected",
        [
            (None, None),
            ({"type": "adaptive"}, None),
            ({"type": "disabled"}, None),
            ({"type": "enabled", "budget_tokens": 1024}, "low"),
            ({"type": "enabled", "budget_tokens": 5000}, "medium"),
            ({"type": "enabled", "budget_tokens": 16000}, "high"),
        ],
    )
    def test_thinking_to_effort(self, thinking, expected):
        assert thinking_to_effort(thinking) == expected

    @pytest.mark.asyncio
    async def test_reasoning_block_from_provider_effort(self):
        agent, fake = _agent([_response([_message_item("hi")])], effort="high")
        await agent.ask_async("hello")
        assert fake.calls[0]["reasoning"] == {"effort": "high", "summary": "auto"}

    @pytest.mark.asyncio
    async def test_chat_thinking_translated_and_not_leaked(self):
        agent, fake = _agent(
            [_response([_message_item("hi")])],
            thinking={"type": "enabled", "budget_tokens": 8000},
        )
        await agent.ask_async("hello")
        call = fake.calls[0]
        assert "thinking" not in call
        assert call["reasoning"]["effort"] == "medium"

    @pytest.mark.asyncio
    async def test_per_call_effort_wins(self):
        agent, fake = _agent([_response([_message_item("hi")])], effort="low")
        await agent.ask_async("hello", effort="high")
        assert fake.calls[0]["reasoning"]["effort"] == "high"
        assert "effort" not in fake.calls[0]

    @pytest.mark.asyncio
    async def test_adaptive_sends_summary_only(self):
        agent, fake = _agent(
            [_response([_message_item("hi")])], thinking={"type": "adaptive"}
        )
        await agent.ask_async("hello")
        assert fake.calls[0]["reasoning"] == {"summary": "auto"}

    @pytest.mark.asyncio
    async def test_request_defaults(self):
        agent, fake = _agent([_response([_message_item("hi")])])
        agent.instruction = "Be terse."
        await agent.ask_async("hello")
        call = fake.calls[0]
        assert call["store"] is False
        assert call["include"] == ["reasoning.encrypted_content"]
        assert call["instructions"] == "Be terse."
        assert call["max_output_tokens"] == agent.provider.max_tokens
        assert "max_tokens" not in call


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_user_text_and_image(self):
        from PIL import Image as PILImage

        image = PILImage.new("RGB", (2, 2))
        image.format = "PNG"
        message = Message(
            role="user",
            parts=[
                MessagePart(type="text", content="look"),
                MessagePart(type="image", image=image),
            ],
        )
        [item] = message_to_responses_items(message)
        assert item["role"] == "user"
        assert item["content"][0] == {"type": "input_text", "text": "look"}
        assert item["content"][1]["type"] == "input_image"
        assert item["content"][1]["image_url"].startswith("data:image/png;base64,")

    def test_pdf_document(self):
        doc = Document(b"%PDF-1.4", name="report.pdf")
        message = Message(
            role="user", parts=[MessagePart(type="document", document=doc)]
        )
        [item] = message_to_responses_items(message)
        assert item["content"][0]["type"] == "input_file"
        assert item["content"][0]["filename"] == "report.pdf"
        assert item["content"][0]["file_data"].startswith(
            "data:application/pdf;base64,"
        )

    def test_text_document_inlined(self):
        doc = Document(b"hello", media_type="text/plain", name="notes.txt")
        message = Message(
            role="user", parts=[MessagePart(type="document", document=doc)]
        )
        [item] = message_to_responses_items(message)
        assert item["content"][0] == {
            "type": "input_text",
            "text": "[Document: notes.txt]\nhello",
        }

    def test_assistant_with_live_reasoning_and_tool_call(self):
        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="thinking",
                    thinking="plan",
                    thinking_signature=encode_thinking_signature("rs_1", "ENC"),
                ),
                MessagePart(type="text", content="Let me check."),
                MessagePart(
                    type="function_call",
                    function_call={"name": "f", "arguments": {"a": 1}},
                    function_call_id="call_1",
                ),
            ],
        )
        message.is_live = True
        items = message_to_responses_items(message)
        assert items[0] == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "encrypted_content": "ENC",
        }
        assert items[1] == {"role": "assistant", "content": "Let me check."}
        assert items[2] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "f",
            "arguments": '{"a": 1}',
        }

    def test_reasoning_dropped_when_not_live(self):
        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="thinking",
                    thinking="plan",
                    thinking_signature=encode_thinking_signature("rs_1", "ENC"),
                ),
                MessagePart(type="text", content="ok"),
            ],
        )
        items = message_to_responses_items(message)
        assert [i.get("type", "message") for i in items] == ["message"]

    def test_anthropic_signature_is_skipped_not_replayed(self):
        # A history produced under Anthropic carries opaque signatures; they
        # must not be sent to OpenAI as encrypted reasoning.
        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="thinking",
                    thinking="plan",
                    thinking_signature="ErUBCkYIBRgC...",
                ),
                MessagePart(type="text", content="ok"),
            ],
        )
        message.is_live = True
        items = message_to_responses_items(message)
        assert all(i.get("type") != "reasoning" for i in items)

    def test_signature_roundtrip(self):
        assert decode_thinking_signature(
            encode_thinking_signature("rs_9", "abc|def")
        ) == ("rs_9", "abc|def")
        assert decode_thinking_signature(None) == (None, None)
        assert decode_thinking_signature("opaque") == (None, None)

    def test_function_results_one_output_per_call(self):
        message = Message(
            role="function",
            parts=[
                MessagePart(
                    type="function_result", content="r1", function_call_id="c1"
                ),
                MessagePart(
                    type="function_result", content="r2", function_call_id="c2"
                ),
            ],
        )
        items = message_to_responses_items(message)
        assert items == [
            {"type": "function_call_output", "call_id": "c1", "output": "r1"},
            {"type": "function_call_output", "call_id": "c2", "output": "r2"},
        ]

    def test_function_result_image_uses_content_array(self):
        from PIL import Image as PILImage

        image = PILImage.new("RGB", (2, 2))
        image.format = "PNG"
        message = Message(
            role="function",
            parts=[
                MessagePart(
                    type="function_result_image",
                    content="chart",
                    image=image,
                    function_call_id="c1",
                ),
            ],
        )
        [item] = message_to_responses_items(message)
        assert item["output"][0] == {"type": "input_text", "text": "chart"}
        assert item["output"][1]["type"] == "input_image"

    def test_compaction_part(self):
        message = Message(
            role="user", parts=[MessagePart(type="compaction", content="summary")]
        )
        [item] = message_to_responses_items(message)
        assert item["content"][0]["text"].startswith("[Previous conversation summary]")


# ---------------------------------------------------------------------------
# Output parsing / usage
# ---------------------------------------------------------------------------


class TestOutputParsing:
    def test_reasoning_text_and_tool_call(self):
        message = output_to_message(
            [_reasoning_item(), _message_item("hello"), _function_call_item()],
            response_id="resp_7",
            usage=_usage(input_tokens=1000, cached=600, output_tokens=50, reasoning=30),
        )
        assert message.is_live is True
        assert message.id == "resp_7"
        assert [p.type for p in message.parts] == ["thinking", "text", "function_call"]
        thinking = message.parts[0]
        assert thinking.thinking == "thinking hard"
        assert thinking.thinking_signature == "rs_1|ENC"
        assert message.parts[2].function_call == {
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
        assert message.parts[2].function_call_id == "call_1"
        assert message.usage == {
            "input_tokens": 400,
            "output_tokens": 50,
            "cache_read_input_tokens": 600,
            "reasoning_tokens": 30,
        }

    def test_reasoning_without_summary_keeps_signature(self):
        message = output_to_message([_reasoning_item(text=""), _message_item("x")])
        assert message.parts[0].thinking is None
        assert message.parts[0].thinking_signature == "rs_1|ENC"

    def test_usage_none(self):
        assert usage_to_dict(None) is None

    def test_usage_dict_input(self):
        assert usage_to_dict({"input_tokens": 10, "output_tokens": 5}) == {
            "input_tokens": 10,
            "output_tokens": 5,
        }


# ---------------------------------------------------------------------------
# Tool loop (non-streaming) with reasoning replay
# ---------------------------------------------------------------------------


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_reasoning_replayed_on_next_turn(self):
        agent, fake = _agent(
            [
                _response([_reasoning_item(), _function_call_item()], response_id="r1"),
                _response(
                    [_reasoning_item("rs_2", "done", "ENC2"), _message_item("21C")],
                    response_id="r2",
                ),
            ]
        )

        def get_weather(city: str) -> str:
            """Weather.

            Args:
                city: the city
            """
            return "21C sunny"

        agent.add_function(get_weather)
        messages = [m async for m in agent.run_conversation_async("Paris?")]
        assert messages[-1].content == "21C"

        second = fake.calls[1]
        types = [i.get("type", "message") for i in second["input"]]
        assert types == [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        ]
        assert second["input"][1]["encrypted_content"] == "ENC"
        assert second["input"][3] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "21C sunny",
        }
        tool = second["tools"][0]
        assert tool["type"] == "function" and tool["name"] == "get_weather"
        assert "defer_loading" not in tool
        # The Responses API defaults strict to true, which rejects schemas
        # with optional arguments.
        assert tool["strict"] is False

    @pytest.mark.asyncio
    async def test_use_tools_only(self):
        agent, fake = _agent([_response([_message_item("x")])], use_tools_only=True)
        await agent.ask_async("go")
        assert fake.calls[0]["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_load_messages_keep_thinking_replays(self):
        agent, fake = _agent([_response([_message_item("x")])])
        history = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                parts=[
                    MessagePart(
                        type="thinking", thinking="t", thinking_signature="rs_1|ENC"
                    ),
                    MessagePart(type="text", content="hello"),
                ],
            ),
        ]
        agent.load_messages(history, keep_thinking=True)
        await agent.ask_async("again")
        types = [i.get("type", "message") for i in fake.calls[0]["input"]]
        assert types == ["message", "reasoning", "message", "message"]

    @pytest.mark.asyncio
    async def test_load_messages_default_strips_thinking(self):
        agent, fake = _agent([_response([_message_item("x")])])
        history = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                parts=[
                    MessagePart(
                        type="thinking", thinking="t", thinking_signature="rs_1|ENC"
                    ),
                    MessagePart(type="text", content="hello"),
                ],
            ),
        ]
        agent.load_messages(history)
        await agent.ask_async("again")
        assert all(i.get("type") != "reasoning" for i in fake.calls[0]["input"])


# ---------------------------------------------------------------------------
# Tool search emulation
# ---------------------------------------------------------------------------


class TestToolSearch:
    @pytest.mark.asyncio
    async def test_deferred_tools_hidden_until_searched(self):
        agent, fake = _agent(
            [
                _response(
                    [
                        _function_call_item(
                            "call_s", "tool_search", '{"query": "weather forecast"}'
                        )
                    ]
                ),
                _response(
                    [_function_call_item("call_w", "get_weather", '{"city": "Rome"}')]
                ),
                _response([_message_item("done")]),
            ]
        )

        def get_weather(city: str) -> str:
            """Get the weather forecast for a city.

            Args:
                city: the city
            """
            return "sunny"

        def get_stock(symbol: str) -> str:
            """Get a stock price.

            Args:
                symbol: ticker
            """
            return "100"

        agent.add_function(get_weather)
        agent.add_function(get_stock)
        agent.enable_tool_search()

        [m async for m in agent.run_conversation_async("Weather in Rome?")]

        first_tools = {t["name"] for t in fake.calls[0]["tools"]}
        assert first_tools == {"tool_search"}
        assert "get_weather" in fake.calls[0]["instructions"]  # categories hint
        second_tools = {t["name"] for t in fake.calls[1]["tools"]}
        assert second_tools == {"tool_search", "get_weather"}
        assert all("defer_loading" not in t for t in fake.calls[1]["tools"])


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _stream_events(final):
    return [
        NS(type="response.created", response=NS(id="resp_s")),
        NS(
            type="response.output_item.added",
            output_index=0,
            item=NS(type="reasoning", id="rs_1"),
        ),
        NS(type="response.reasoning_summary_text.delta", delta="thin", output_index=0),
        NS(type="response.reasoning_summary_text.delta", delta="king", output_index=0),
        NS(
            type="response.output_item.added",
            output_index=1,
            item=NS(type="message", id="msg_1"),
        ),
        NS(type="response.output_text.delta", delta="Hel", output_index=1),
        NS(type="response.output_text.delta", delta="lo", output_index=1),
        NS(
            type="response.output_item.added",
            output_index=2,
            item=_function_call_item("call_1", "get_weather", ""),
        ),
        NS(
            type="response.function_call_arguments.delta",
            delta='{"city":',
            output_index=2,
        ),
        NS(
            type="response.function_call_arguments.delta",
            delta=' "Paris"}',
            output_index=2,
        ),
        NS(type="response.completed", response=final),
    ]


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_events_and_final_message(self):
        final = _response(
            [
                _reasoning_item(text="thinking"),
                _message_item("Hello"),
                _function_call_item(),
            ],
            response_id="resp_s",
            usage=_usage(input_tokens=50, output_tokens=10, reasoning=4),
        )
        agent, fake = _agent([_stream_events(final)])
        chunks = [c async for c in agent.ask_stream_async("hi")]
        assert fake.calls[0]["stream"] is True

        assert [c["content"] for c in chunks if c["type"] == "thinking"] == [
            "thin",
            "king",
        ]
        assert [c["content"] for c in chunks if c["type"] == "text"] == ["Hel", "lo"]
        started = [c for c in chunks if c["type"] == "tool_started"]
        assert started == [
            {"type": "tool_started", "name": "get_weather", "id": "call_1", "index": 2}
        ]
        deltas = [c["partial_json"] for c in chunks if c["type"] == "tool_delta"]
        assert "".join(deltas) == '{"city": "Paris"}'
        assert all(
            c["name"] == "get_weather" for c in chunks if c["type"] == "tool_delta"
        )

        final_message = chunks[-1]["message"]
        assert [p.type for p in final_message.parts] == [
            "thinking",
            "text",
            "function_call",
        ]
        assert final_message.usage["reasoning_tokens"] == 4
        # Usage arrives with the completed event, so compaction can see it.
        assert agent.messages[-1].usage["input_tokens"] == 50

    @pytest.mark.asyncio
    async def test_stream_failure_raises(self):
        failed = NS(id="r", output=[], usage=None, error=NS(message="boom"))
        agent, fake = _agent([[NS(type="response.failed", response=failed)]])
        with pytest.raises(RuntimeError, match="boom"):
            [c async for c in agent.ask_stream_async("hi")]


# ---------------------------------------------------------------------------
# complete() / compaction
# ---------------------------------------------------------------------------


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_uses_instructions_and_max_output_tokens(self):
        agent, fake = _agent([_response([_message_item("summary text")])])
        text = await agent.provider.complete(
            [{"role": "user", "content": "summarize"}],
            system="You summarize.",
            max_tokens=512,
        )
        assert text == "summary text"
        call = fake.calls[0]
        assert call["instructions"] == "You summarize."
        assert call["max_output_tokens"] == 512
        assert call["store"] is False
        assert "max_tokens" not in call

    @pytest.mark.asyncio
    async def test_compaction_triggers_from_streamed_usage(self):
        from agentlys.compaction import TokenThresholdCompaction

        big = _response([_message_item("first")], usage=_usage(input_tokens=5000))
        summary = _response([_message_item("the summary")])
        after = _response([_message_item("second")])
        agent, fake = _agent(
            [_stream_events(big), summary, _stream_events(after)],
            compaction=TokenThresholdCompaction(token_threshold=1000),
        )
        [c async for c in agent.ask_stream_async("one")]
        chunks = [c async for c in agent.ask_stream_async("two")]
        assert any(c["type"] == "compacting" for c in chunks)
        assert agent.messages[0].has_compaction


class TestReplayEdgeCases:
    def test_trailing_reasoning_is_not_replayed(self):
        # A turn cut off by max_output_tokens holds reasoning and nothing
        # else; the API rejects a reasoning item without its following item.
        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="thinking", thinking="t", thinking_signature="rs_1|ENC"
                ),
                MessagePart(type="text", content=""),
            ],
        )
        message.is_live = True
        assert message_to_responses_items(message) == []

    def test_reasoning_without_id_is_skipped(self):
        message = Message(
            role="assistant",
            parts=[
                MessagePart(type="thinking", thinking="t", thinking_signature="|ENC"),
                MessagePart(type="text", content="ok"),
            ],
        )
        message.is_live = True
        assert message_to_responses_items(message) == [
            {"role": "assistant", "content": "ok"}
        ]

    def test_refusal_becomes_text(self):
        item = NS(
            type="message",
            content=[NS(type="refusal", refusal="I can't help with that.")],
        )
        message = output_to_message([item])
        assert message.content == "I can't help with that."

    @pytest.mark.asyncio
    async def test_stream_joins_summary_parts_like_the_stored_message(self):
        final = _response([_reasoning_item(text="a\n\nb"), _message_item("x")])
        events = [
            NS(
                type="response.output_item.added",
                output_index=0,
                item=NS(type="reasoning"),
            ),
            NS(type="response.reasoning_summary_part.added", output_index=0),
            NS(type="response.reasoning_summary_text.delta", delta="a", output_index=0),
            NS(type="response.reasoning_summary_part.added", output_index=0),
            NS(type="response.reasoning_summary_text.delta", delta="b", output_index=0),
            NS(type="response.completed", response=final),
        ]
        agent, fake = _agent([events])
        chunks = [c async for c in agent.ask_stream_async("hi")]
        streamed = "".join(c["content"] for c in chunks if c["type"] == "thinking")
        assert streamed == chunks[-1]["message"].parts[0].thinking == "a\n\nb"

    @pytest.mark.asyncio
    async def test_incomplete_response_is_logged(self, caplog):
        final = _response([_reasoning_item()])
        final.status = "incomplete"
        final.incomplete_details = NS(reason="max_output_tokens")
        agent, fake = _agent([final])
        with caplog.at_level("WARNING"):
            await agent.ask_async("hi")
        assert "max_output_tokens" in caplog.text
