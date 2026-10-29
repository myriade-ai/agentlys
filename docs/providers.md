# Provider Guide

Agentlys supports multiple LLM providers out of the box, with easy configuration and the ability to create custom providers.

## Supported Providers

### OpenAI

OpenAI's GPT models are the default provider.

**Setup:**

```bash
export OPENAI_API_KEY="your-openai-key"
export AGENTLYS_MODEL="gpt-5-mini"  # optional, this is the default
```

**Usage:**

```python
from agentlys import Agentlys

# Default (uses OpenAI)
agent = Agentlys()

# Explicit OpenAI
agent = Agentlys(provider="openai")

# Specific model
agent = Agentlys(provider="openai", model="gpt-5-mini")
```

### OpenAI Responses API (GPT-5 / o-series reasoning)

The `openai_responses` provider targets the Responses API, which is the only
OpenAI API that carries a reasoning model's hidden state across the turns of
a tool loop. Use it for GPT-5 / o-series models; the plain `openai`
provider (Chat Completions) restarts the reasoning from scratch after every
tool result.

```python
agent = Agentlys(provider="openai_responses", model="gpt-5.4", effort="high")
```

- **Reasoning** — `effort` accepts the OpenAI levels (`none`, `minimal`,
  `low`, `medium`, `high`, `xhigh`) plus Anthropic's `max` (mapped to
  `xhigh`), so one setting works across providers; an Anthropic-style
  `thinking=` config is translated (`enabled` scales with `budget_tokens`,
  `adaptive` leaves the model's default). A summary is requested
  (`reasoning.summary="auto"`) and streamed as `{"type": "thinking"}` chunks.
- **Round-trip** — each turn's reasoning is stored as a `thinking`
  MessagePart whose `thinking_signature` is
  `"<reasoning item id>|<encrypted_content>"`. Persist `thinking` and
  `thinking_signature` byte-for-byte and reload with
  `load_messages(keep_thinking=True)` exactly as for Anthropic; requests use
  `store=false`, nothing is retained on OpenAI's side.
- **Streaming** — yields `text`, `thinking`, `tool_started` (`name`, `id`,
  `index`) and `tool_delta` (`partial_json`) chunks before the final
  `message`. Usage (including `cache_read_input_tokens` and
  `reasoning_tokens`) arrives with the completed response, so compaction
  works in streaming mode.
- **Tool search** — agentlys keeps its own client-side search tool rather
  than the API's native `tool_search`; deferred tools are hidden until a
  search result references them, the same economy as Anthropic's
  `defer_loading` (the tool list changes when a tool loads, which resets
  OpenAI's automatic prefix cache for that request).
- `base_url`, `api_key`, `AGENTLYS_HOST`, `AGENTLYS_API_KEY` and
  `default_headers` (constructor only) work as for the `openai` provider.
  `cache_ttl` / `cache_ttl_messages` are accepted and ignored — OpenAI
  caches prompt prefixes automatically.

The `openai` provider also maps `effort` / `thinking` to `reasoning_effort`,
hides deferred tools the same way, requests usage in streams
(`AGENTLYS_OPENAI_STREAM_USAGE=0` opts out) and uses `max_completion_tokens`
(`AGENTLYS_OPENAI_LEGACY_MAX_TOKENS=1` keeps `max_tokens` for gateways that
reject the new field), so reasoning models work there too — without the
cross-turn reasoning state.

### Anthropic

Anthropic's Claude models, recommended for agentic behavior.

**Setup:**

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export AGENTLYS_MODEL="claude-sonnet-4-20250514"
```

**Usage:**

```python
# Use Anthropic
agent = Agentlys(provider="anthropic")

# Specific Claude model
agent = Agentlys(provider="anthropic", model="claude-sonnet-4-20250514")
```

### Any OpenAI-Compatible API

The `openai` provider works with any API that speaks the OpenAI chat
completions protocol: Ollama, vLLM, LiteLLM, OpenRouter, Together, Groq,
Azure-hosted gateways, self-hosted models, etc. Point it at your endpoint
with `base_url` (and `api_key` if the endpoint requires one):

```python
# Ollama running locally (no API key needed)
agent = Agentlys(
    provider="openai",
    model="llama3.1",
    base_url="http://localhost:11434/v1",
)

# OpenRouter
agent = Agentlys(
    provider="openai",
    model="meta-llama/llama-3.1-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
)
```

Or configure everything through environment variables:

```bash
export AGENTLYS_PROVIDER="openai"
export AGENTLYS_HOST="http://localhost:11434/v1"
export AGENTLYS_API_KEY="optional-key"   # falls back to OPENAI_API_KEY
export AGENTLYS_MODEL="llama3.1"
```

```python
agent = Agentlys()  # picks everything up from the environment
```

Resolution order: explicit arguments > `AGENTLYS_*` env vars > provider
defaults. When a custom endpoint is configured without any API key, a
placeholder key is sent so key-less servers (Ollama, vLLM, ...) work out
of the box.

### Custom Provider Host

Use alternative endpoints or self-hosted models:

```bash
export AGENTLYS_HOST="https://your-custom-endpoint.com"
```

## Configuration Patterns

### Environment-Based Configuration

The simplest approach using environment variables:

```python
# .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
AGENTLYS_MODEL=claude-sonnet-4-20250514

# Python code
from agentlys import Agentlys

# Uses environment variables automatically
agent = Agentlys(provider="anthropic")
```

### Runtime Configuration

Configure providers programmatically:

```python
from agentlys import Agentlys

# OpenAI with custom settings
openai_agent = Agentlys(
    provider="openai",
    model="gpt-5-mini",
    instruction="You are a helpful assistant"
)

# Anthropic with custom settings
claude_agent = Agentlys(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    instruction="You are a thoughtful assistant"
)
```

### Multi-Provider Setup

Use different providers for different tasks:

```python
class AgentManager:
    def __init__(self):
        # Fast agent for simple tasks
        self.quick_agent = Agentlys(
            provider="openai",
            model="gpt-5-mini",
            instruction="Provide quick, concise answers"
        )

        # Powerful agent for complex tasks
        self.smart_agent = Agentlys(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            instruction="Think deeply and provide comprehensive solutions"
        )

    def route_request(self, query: str, complexity: str = "simple"):
        if complexity == "simple":
            return self.quick_agent.ask(query)
        else:
            return self.smart_agent.ask(query)
```

## Custom Providers

Create custom providers for specialized use cases:

### Basic Custom Provider

```python
from agentlys.providers.base_provider import BaseProvider
from agentlys import Agentlys, Message

class MyCustomProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = kwargs.get('model', 'my-default-model')

    def create_completion(self, messages, **kwargs):
        # Implement your provider's API call
        # This should return a Message object

        # Example implementation:
        response_text = self._call_my_api(messages)

        return Message(
            role="assistant",
            content=response_text
        )

    def _call_my_api(self, messages):
        # Your custom API integration logic
        import requests

        response = requests.post(
            "https://my-llm-api.com/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [msg.to_dict() for msg in messages]
            }
        )

        return response.json()["choices"][0]["message"]["content"]

# Usage
custom_agent = Agentlys(provider=MyCustomProvider(api_key="your-key"))
```

### Advanced Custom Provider with Function Calling

```python
from agentlys.providers.base_provider import BaseProvider
from agentlys import Message, MessagePart
import json

class AdvancedCustomProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_completion(self, messages, functions=None, **kwargs):
        # Convert agentlys messages to your provider's format
        provider_messages = self._convert_messages(messages)

        # Include function schemas if provided
        payload = {
            "messages": provider_messages,
            "model": self.model
        }

        if functions:
            payload["functions"] = [self._convert_function_schema(f) for f in functions]

        # Call your provider's API
        response = self._api_call(payload)

        # Handle function calls in response
        if "function_call" in response:
            return Message(
                role="assistant",
                function_call=response["function_call"]
            )
        else:
            return Message(
                role="assistant",
                content=response["content"]
            )

    def _convert_messages(self, messages):
        # Convert agentlys Message objects to your provider's format
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.role,
                "content": msg.content
            })
        return converted

    def _convert_function_schema(self, function_schema):
        # Convert agentlys function schema to your provider's format
        return {
            "name": function_schema["name"],
            "description": function_schema["description"],
            "parameters": function_schema["parameters"]
        }

    def _api_call(self, payload):
        # Your API call implementation
        pass
```

## Provider-Specific Features

### OpenAI Features

```python
# OpenAI-specific model configurations
openai_agent = Agentlys(
    provider="openai",
    model="gpt-5-mini",
    # OpenAI-specific parameters can be passed through kwargs
)

# Access to OpenAI's latest models
o1_agent = Agentlys(
    provider="openai",
    model="o1-preview"  # For complex reasoning tasks
)
```

### Anthropic Features

```python
# Anthropic excels at tool use and following instructions
anthropic_agent = Agentlys(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    instruction="""You are a meticulous developer agent that:
    1. Always follows best practices
    2. Writes comprehensive tests
    3. Documents code thoroughly
    4. Considers edge cases

    Use the available tools systematically."""
)
```

### Prompt Caching, Effort and Debugging (Anthropic)

The Anthropic provider places its four cache breakpoints on the last system
block, the last non-deferred tool, `messages[-3]` and `messages[-1]`. The TTL
of those breakpoints, the reasoning effort and a cache debug trace are
configurable — every one of them is off/default unless set:

| Setting | Env var | Constructor | Values | Default |
|---|---|---|---|---|
| System + tools cache TTL | `AGENTLYS_CACHE_TTL` | `cache_ttl` | `5m`, `1h` | `5m` |
| Message cache TTL | `AGENTLYS_CACHE_TTL_MESSAGES` | `cache_ttl_messages` | `5m`, `1h` | same as `cache_ttl` |
| 1h-cache beta header | `AGENTLYS_CACHE_TTL_BETA` | — | `1`, `0` | `1` (sent only when a 1h TTL is used) |
| Reasoning effort | `AGENTLYS_EFFORT` | `effort` | `low` … `max` | not sent |
| Cache debug logs | `AGENTLYS_CACHE_DEBUG` | — | `1` | off |

```python
agent = Agentlys(
    provider="anthropic",
    model="claude-opus-5",
    cache_ttl="1h",           # system + tools survive a 20-minute pause
    cache_ttl_messages="5m",  # the conversation tail turns over fast anyway
    effort="medium",          # per-call override: agent.ask(msg, effort="max")
)
```

A 1-hour write costs 2x base input against 1.25x for 5 minutes, so it pays
off only when requests sharing the prefix are more than 5 minutes apart — that
is why the system/tools and message breakpoints are configured separately. The
API also requires longer-lived entries to come first, so a 1h message TTL
under a 5m system TTL is refused and clamped back to 5m with a warning.

### Keeping the cache across turns (`load_messages`)

Caching is a prefix match across turns too, not only within one tool loop. An
assistant message that was *sent* with its thinking blocks and is later
reloaded without them changes the prefix at that position, so the first call of
each new question rewrites the whole conversation instead of reading it from
cache. If your storage keeps `thinking` and `thinking_signature` verbatim, in
their original order, opt in:

```python
agent.load_messages(messages, keep_thinking=True)
```

Blocks that lost their signature are dropped either way — the API rejects
those. No model check is needed: regular thinking blocks are not origin-locked,
the server renders them into the target model's prompt. The default stays
`False`, which strips thinking from every reloaded assistant message.

`AGENTLYS_CACHE_DEBUG=1` logs, at INFO on `agentlys.providers.anthropic`, the
sha256 of the system blocks, of the tools array and of each message, plus the
usage token counts of the response — enough to spot which section of the
prefix drifted between two requests without patching the library.

## Error Handling

Handle provider-specific errors gracefully:

```python
from agentlys import Agentlys
from openai import RateLimitError
from anthropic import AuthenticationError

def robust_agent_call(query: str):
    agents = [
        Agentlys(provider="anthropic"),
        Agentlys(provider="openai", model="gpt-5-mini"),  # Fallback
    ]

    for agent in agents:
        try:
            return agent.ask(query)
        except (RateLimitError, AuthenticationError) as e:
            print(f"Provider error: {e}, trying next provider...")
            continue

    raise Exception("All providers failed")
```
