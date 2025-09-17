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
