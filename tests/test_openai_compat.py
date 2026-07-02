"""Tests for using any OpenAI-compatible API (Ollama, vLLM, LiteLLM, ...).

These tests never hit the network: they only check client configuration
and the parsing/serialization of OpenAI-style payloads.
"""

import json
from types import SimpleNamespace

import pytest
from agentlys import Agentlys
from agentlys.model import Message, MessagePart
from agentlys.providers.openai import (
    from_openai_object,
    message_to_openai_dict,
    split_function_results,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AGENTLYS_HOST", "AGENTLYS_API_KEY", "AGENTLYS_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")


class TestClientConfiguration:
    def test_default_base_url(self):
        agent = Agentlys(provider="openai", model="gpt-4o")
        assert str(agent.provider.client.base_url) == "https://api.openai.com/v1/"
        assert agent.provider.client.api_key == "sk-from-env"

    def test_explicit_base_url_and_api_key(self):
        agent = Agentlys(
            provider="openai",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        assert str(agent.provider.client.base_url) == "http://localhost:11434/v1/"
        assert agent.provider.client.api_key == "ollama"

    def test_env_base_url_and_api_key(self, monkeypatch):
        monkeypatch.setenv("AGENTLYS_HOST", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("AGENTLYS_API_KEY", "sk-or-123")
        agent = Agentlys(provider="openai", model="some-model")
        assert str(agent.provider.client.base_url) == "https://openrouter.ai/api/v1/"
        assert agent.provider.client.api_key == "sk-or-123"

    def test_explicit_arguments_beat_env(self, monkeypatch):
        monkeypatch.setenv("AGENTLYS_HOST", "https://env-host.example/v1")
        monkeypatch.setenv("AGENTLYS_API_KEY", "env-key")
        agent = Agentlys(
            provider="openai",
            model="m",
            base_url="http://localhost:8000/v1",
            api_key="arg-key",
        )
        assert str(agent.provider.client.base_url) == "http://localhost:8000/v1/"
        assert agent.provider.client.api_key == "arg-key"

    def test_keyless_custom_endpoint_gets_placeholder_key(self, monkeypatch):
        # Key-less local servers (Ollama, vLLM, ...) must work without any
        # API key configured — the SDK refuses to build a client without one.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = Agentlys(
            provider="openai", model="llama3.1", base_url="http://localhost:11434/v1"
        )
        assert agent.provider.client.api_key == "not-needed"

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENTLYS_PROVIDER", "openai")
        monkeypatch.setenv("AGENTLYS_HOST", "http://localhost:4000")
        agent = Agentlys(model="my-model")
        assert agent.provider.__class__.__name__ == "OpenAIProvider"
        assert (
            str(agent.provider.client.base_url).rstrip("/") == "http://localhost:4000"
        )

    def test_default_provider_honors_base_url(self):
        agent = Agentlys(
            provider="default",
            model="m",
            base_url="http://localhost:1234/v1",
            api_key="k",
        )
        assert str(agent.provider.client.base_url) == "http://localhost:1234/v1/"
        assert agent.provider.client.api_key == "k"


def _tool_call(id, name, arguments):
    return SimpleNamespace(
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestParallelToolCalls:
    def test_from_openai_object_keeps_all_tool_calls(self):
        message = from_openai_object(
            role="assistant",
            content=None,
            tool_calls=[
                _tool_call("call_1", "get_weather", '{"city": "Paris"}'),
                _tool_call("call_2", "get_time", '{"tz": "CET"}'),
            ],
            id="resp_1",
        )
        assert len(message.function_call_parts) == 2
        assert message.function_call_parts[0].function_call["name"] == "get_weather"
        assert message.function_call_parts[1].function_call_id == "call_2"

    def test_message_to_openai_dict_serializes_all_tool_calls(self):
        message = Message(
            role="assistant",
            parts=[
                MessagePart(
                    type="function_call",
                    function_call={"name": "f1", "arguments": {"a": 1}},
                    function_call_id="call_1",
                ),
                MessagePart(
                    type="function_call",
                    function_call={"name": "f2", "arguments": {"b": 2}},
                    function_call_id="call_2",
                ),
            ],
        )
        res = message_to_openai_dict(message)
        assert [t["id"] for t in res["tool_calls"]] == ["call_1", "call_2"]

    def test_split_function_results_one_tool_message_per_call(self):
        combined = Message(
            role="function",
            parts=[
                MessagePart(
                    type="function_result", content="r1", function_call_id="call_1"
                ),
                MessagePart(
                    type="function_result", content="r2", function_call_id="call_2"
                ),
            ],
        )
        split = split_function_results([combined])
        assert len(split) == 2
        dicts = [message_to_openai_dict(m) for m in split]
        assert dicts[0]["tool_call_id"] == "call_1"
        assert dicts[1]["tool_call_id"] == "call_2"

    def test_split_function_results_keeps_single_result_untouched(self):
        single = Message(
            role="function",
            name="f1",
            parts=[
                MessagePart(
                    type="function_result", content="r1", function_call_id="call_1"
                )
            ],
        )
        assert split_function_results([single]) == [single]


def _stream_chunk(id="resp_1", delta=None, usage=None):
    choices = []
    if delta is not None:
        choices = [SimpleNamespace(delta=delta)]
    return SimpleNamespace(id=id, choices=choices, usage=usage)


def _delta(content=None, role=None, tool_calls=None):
    return SimpleNamespace(content=content, role=role, tool_calls=tool_calls)


def _delta_tool_call(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeStreamClient:
    """Minimal stand-in for AsyncOpenAI limited to streaming completions."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None

        async def create(**kwargs):
            self.kwargs = kwargs

            async def iterator():
                for chunk in chunks:
                    yield chunk

            return iterator()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.mark.asyncio
async def test_fetch_stream_async_text():
    agent = Agentlys(provider="openai", model="gpt-4o")
    agent.messages.append(Message(role="user", content="Hello"))
    client = _FakeStreamClient(
        [
            _stream_chunk(delta=_delta(role="assistant", content="Hel")),
            _stream_chunk(delta=_delta(content="lo!")),
            _stream_chunk(
                delta=None,
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            ),
        ]
    )
    agent.provider.client = client

    events = [event async for event in agent.provider.fetch_stream_async()]

    assert client.kwargs["stream"] is True
    text_events = [e for e in events if e["type"] == "text"]
    assert [e["content"] for e in text_events] == ["Hel", "lo!"]
    final = events[-1]
    assert final["type"] == "message"
    assert final["message"].content == "Hello!"
    assert final["message"].usage == {"input_tokens": 5, "output_tokens": 2}


@pytest.mark.asyncio
async def test_fetch_stream_async_accumulates_tool_calls():
    agent = Agentlys(provider="openai", model="gpt-4o")
    agent.messages.append(Message(role="user", content="Weather in Paris and Lyon?"))
    client = _FakeStreamClient(
        [
            _stream_chunk(
                delta=_delta(
                    role="assistant",
                    tool_calls=[
                        _delta_tool_call(0, id="call_1", name="get_weather"),
                    ],
                )
            ),
            _stream_chunk(
                delta=_delta(
                    tool_calls=[
                        _delta_tool_call(0, arguments='{"city": '),
                        _delta_tool_call(0, arguments='"Paris"}'),
                        _delta_tool_call(
                            1,
                            id="call_2",
                            name="get_weather",
                            arguments='{"city": "Lyon"}',
                        ),
                    ]
                )
            ),
        ]
    )
    agent.provider.client = client

    events = [event async for event in agent.provider.fetch_stream_async()]

    final_message = events[-1]["message"]
    parts = final_message.function_call_parts
    assert len(parts) == 2
    assert parts[0].function_call == {
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }
    assert parts[0].function_call_id == "call_1"
    assert parts[1].function_call["arguments"] == {"city": "Lyon"}


@pytest.mark.asyncio
async def test_fetch_stream_async_sends_tools():
    agent = Agentlys(provider="openai", model="gpt-4o")

    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return "sunny"

    agent.add_function(get_weather)
    agent.messages.append(Message(role="user", content="Weather in Paris?"))
    client = _FakeStreamClient(
        [_stream_chunk(delta=_delta(role="assistant", content="Sunny"))]
    )
    agent.provider.client = client

    async for _ in agent.provider.fetch_stream_async():
        pass

    tool_names = [t["function"]["name"] for t in client.kwargs["tools"]]
    assert tool_names == ["get_weather"]


@pytest.mark.asyncio
async def test_run_conversation_stream_async_end_to_end():
    """The full streaming tool loop works against an OpenAI-compatible client."""
    agent = Agentlys(
        provider="openai",
        model="llama3.1",
        base_url="http://localhost:11434/v1",
        instruction="You are a helpful assistant",
    )

    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return "sunny"

    agent.add_function(get_weather)

    call_count = {"n": 0}

    async def create(**kwargs):
        call_count["n"] += 1

        if call_count["n"] == 1:
            chunks = [
                _stream_chunk(
                    delta=_delta(
                        role="assistant",
                        tool_calls=[
                            _delta_tool_call(
                                0,
                                id="call_1",
                                name="get_weather",
                                arguments=json.dumps({"city": "Paris"}),
                            )
                        ],
                    )
                )
            ]
        else:
            chunks = [
                _stream_chunk(delta=_delta(role="assistant", content="It is sunny!"))
            ]

        async def iterator():
            for chunk in chunks:
                yield chunk

        return iterator()

    agent.provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    events = [
        event
        async for event in agent.run_conversation_stream_async("Weather in Paris?")
    ]

    types = [e["type"] for e in events]
    assert "tool_result" in types
    assistant_messages = [e["message"] for e in events if e["type"] == "assistant"]
    assert assistant_messages[-1].content == "It is sunny!"
    # The tool result was sent back as its own tool message on round 2
    assert call_count["n"] == 2


class TestCompleteAndCompaction:
    """complete() and compaction against OpenAI-compatible providers."""

    def _agent_with_mocked_completion(self, text):
        from unittest.mock import AsyncMock

        agent = Agentlys(
            provider="openai",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            instruction="You are a helpful assistant",
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )
        mock_create = AsyncMock(return_value=response)
        agent.provider.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
        )
        return agent, mock_create

    @pytest.mark.asyncio
    async def test_complete_prepends_system_and_defaults_model(self):
        agent, mock_create = self._agent_with_mocked_completion("a summary")

        result = await agent.provider.complete(
            messages=[{"role": "user", "content": "Summarize this"}],
            system="You summarize conversations",
        )

        assert result == "a summary"
        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "llama3.1"  # falls back to the provider model
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": "You summarize conversations",
        }

    @pytest.mark.asyncio
    async def test_compaction_works_with_openai_provider(self):
        from agentlys.compaction import TokenThresholdCompaction

        agent, mock_create = self._agent_with_mocked_completion(
            "<summary>Talked about the weather</summary>"
        )
        agent.messages = [
            Message(role="user", content="First message"),
            Message(role="assistant", content="First response"),
            Message(role="user", content="Second message"),
            Message(role="assistant", content="Second response"),
        ]

        compaction = TokenThresholdCompaction(token_threshold=100)
        await compaction.compact(agent)

        assert len(agent.messages) == 1
        assert agent.messages[0].has_compaction
        assert agent.messages[0].parts[0].content == "Talked about the weather"
        # The agent instruction rides along as the system message
        kwargs = mock_create.call_args.kwargs
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_complete_not_implemented_on_bare_provider(self):
        from agentlys.providers.base_provider import BaseProvider

        class Bare(BaseProvider):
            def __init__(self, chat, model):
                self.chat = chat
                self.model = model

            async def fetch_async(self, **kwargs):
                return Message(role="assistant", content="hi")

        agent = Agentlys(provider=Bare, model="m")
        with pytest.raises(NotImplementedError):
            await agent.provider.complete(messages=[{"role": "user", "content": "x"}])


class TestDefaultProviderTransform:
    """DefaultProvider must apply its string-only wire format."""

    def _agent(self):
        return Agentlys(
            provider="default",
            model="m",
            base_url="http://localhost:1234/v1",
            instruction="Be brief",
        )

    def test_system_messages_are_strings(self):
        agent = self._agent()
        agent.messages = [Message(role="user", content="Hello")]
        messages, _, _ = agent.provider._prepare_request_params()
        assert messages[0] == {"role": "system", "content": "Be brief"}

    def test_tool_results_are_strings(self):
        agent = self._agent()
        agent.messages = [
            Message(role="user", content="Weather?"),
            Message(
                role="assistant",
                parts=[
                    MessagePart(
                        type="function_call",
                        function_call={"name": "get_weather", "arguments": {}},
                        function_call_id="call_1",
                    )
                ],
            ),
            Message(
                role="function",
                name="get_weather",
                parts=[
                    MessagePart(
                        type="function_result",
                        content="sunny",
                        function_call_id="call_1",
                    )
                ],
            ),
        ]
        messages, _, _ = agent.provider._prepare_request_params()
        tool_message = next(m for m in messages if m["role"] == "tool")
        assert tool_message["content"] == "sunny"
        assert tool_message["tool_call_id"] == "call_1"


def test_compaction_part_serializes_as_text():
    """Post-compaction history must serialize on the OpenAI wire format."""
    from agentlys.providers.openai import parts_to_openai_dict

    part = MessagePart(type="compaction", content="Talked about the weather")
    result = parts_to_openai_dict(part)
    assert result["type"] == "text"
    assert "[Previous conversation summary]" in result["text"]
    assert "Talked about the weather" in result["text"]
