from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentlys import Agentlys
from agentlys.model import Message
from agentlys.providers.base_provider import APIProvider
from agentlys.providers.openai import from_openai_object, message_to_openai_dict
from agentlys.providers.utils import FunctionCallParsingError

RAW = r"""{"text": "l\'outil"}"""


def response(raw=RAW, call_id="call_bad"):
    return from_openai_object(
        "assistant",
        None,
        [
            SimpleNamespace(
                type="function",
                id=call_id,
                function=SimpleNamespace(name="save", arguments=raw),
            )
        ],
        id="completion",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


@pytest.mark.parametrize("raw", [RAW, "{", "[]", "null"])
def test_invalid_arguments_preserved(raw):
    msg = response(raw)
    part = msg.function_call_parts[0]
    assert part.function_call["arguments"] == {}
    assert part.arguments_parsing_error
    assert part.raw_arguments == raw
    assert msg.usage["output_tokens"] == 5
    assert message_to_openai_dict(msg)["tool_calls"][0]["function"]["arguments"] == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("exhaust", [False, True])
async def test_reask_and_budget(stream, exhaust):
    agent = Agentlys(provider=APIProvider.OPENAI, api_key="test")
    saved = []

    def save(text: str):
        """Save text."""
        saved.append(text)
        return "saved"

    agent.add_function(save)
    replies = [response(), response()]
    if exhaust:
        replies.append(response())
    else:
        replies.extend(
            [response('{"text": "correct"}'), Message(role="assistant", content="done")]
        )
    pending = iter(replies)

    async def fetch(**kwargs):
        return next(pending)

    async def fetch_stream(**kwargs):
        yield {"type": "message", "message": next(pending)}

    async def run():
        if stream:
            return [item async for item in agent.run_conversation_stream_async("go")]
        return [item async for item in agent.run_conversation_async("go")]

    with (
        patch.object(agent.provider, "fetch_async", side_effect=fetch),
        patch.object(agent.provider, "fetch_stream_async", side_effect=fetch_stream),
    ):
        if exhaust:
            with pytest.raises(FunctionCallParsingError):
                await run()
        else:
            await run()

    assert saved == ([] if exhaust else ["correct"])
    failures = [
        m
        for m in agent.messages
        if m.role == "function" and "Invalid JSON" in (m.content or "")
    ]
    assert len(failures) == 2
    assert RAW in failures[0].content
    assert "Re-emit" in failures[0].content
    assert sum(m.usage["output_tokens"] for m in agent.messages if m.usage) == 15


@pytest.mark.asyncio
async def test_stream_provider_preserves_malformed_fragments_and_usage():
    agent = Agentlys(provider=APIProvider.OPENAI, api_key="test")

    async def chunks():
        for fragment in [RAW[:12], RAW[12:]]:
            yield SimpleNamespace(
                id="completion",
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_bad",
                                    function=SimpleNamespace(
                                        name="save" if fragment == RAW[:12] else None,
                                        arguments=fragment,
                                    ),
                                )
                            ],
                        )
                    )
                ],
            )
        yield SimpleNamespace(
            id="completion",
            choices=[],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    with patch.object(
        agent.provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=chunks(),
    ):
        events = [event async for event in agent.provider.fetch_stream_async()]
    msg = events[-1]["message"]
    assert msg.function_call_parts[0].raw_arguments == RAW
    assert msg.function_call_parts[0].arguments_parsing_error
    assert msg.usage == {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.asyncio
async def test_parallel_batch_skips_only_invalid_call():
    agent = Agentlys(provider=APIProvider.OPENAI, api_key="test")
    saved = []

    def save(text: str):
        """Save text."""
        saved.append(text)
        return "saved"

    agent.add_function(save)
    msg = response()
    msg.parts.extend(response('{"text": "valid"}', "call_good").parts)
    result = await agent._call_functions_parallel(msg.function_call_parts, msg)
    assert saved == ["valid"]
    assert {p.function_call_id for p in result.parts} == {"call_bad", "call_good"}
    assert agent._check_parsing_retry_budget(msg, 0) == 1
