# API Reference

## Agentlys Class

### Constructor

```python
Agentlys(
    name: str = None,
    instruction: str = None,
    examples: list[Message] = None,
    messages: list[Message] = None,
    context: str = None,
    max_interactions: int = 100,
    model: str = None,
    provider: str | Type[BaseProvider] = APIProvider.OPENAI,
    use_tools_only: bool = False,
    mcp_servers: list[object] = []
)
```

**Parameters:**

- `name`: Optional name for the agent (useful when using agent as a tool)
- `instruction`: System instruction that defines the agent's behavior
- `examples`: List of example messages for few-shot learning
- `messages`: Initial conversation history
- `context`: Additional context for the agent
- `max_interactions`: Maximum number of interactions in a conversation (default: 100)
- `model`: Model to use (defaults to env var AGENTLYS_MODEL)
- `provider`: LLM provider ("openai", "anthropic", or custom provider class)
- `use_tools_only`: Beta feature - agent only uses tools, no LLM calls
- `mcp_servers`: List of MCP server connections

### Core Methods

#### ask(message) -> Message

Send a single message and get a response.

```python
response = agent.ask("What is Python?")
print(response.content)
```

#### ask_async(message) -> Message

Async version of ask().

```python
response = await agent.ask_async("What is Python?")
```

#### run_conversation(prompt) -> Generator[Message]

Run a conversation as a generator, yielding each message.

```python
for message in agent.run_conversation("Help me code a web server"):
    print(message.to_markdown())
```

#### run_conversation_async(prompt) -> AsyncGenerator[Message]

Async version of run_conversation().

```python
async for message in agent.run_conversation_async("Help me code"):
    print(message.to_markdown())
```

### Tool Management

#### add_function(func, function_schema=None, defer_loading=False)

Add a Python function as a tool.

```python
def calculate(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

agent.add_function(calculate)

# Defer a tool so it's hidden from the LLM until discovered via tool search
agent.add_function(calculate, defer_loading=True)
```

#### add_tool(obj, name=None)

Add a Python object/class instance as a tool.

```python
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

calc = Calculator()
agent.add_tool(calc, "Calculator")
```

### Tool Search

#### enable_tool_search(always_loaded=None, search_fn=None, search_model=None)

Enable tool search to defer most tools and discover them on-demand. Reduces context window usage and improves tool selection accuracy for agents with many tools.

When enabled, all registered tools (except those in `always_loaded`) are marked with `defer_loading=True`. A `tool_search` function is automatically registered that the LLM can call to discover relevant tools. Tools added after this call are also auto-deferred.

```python
agent.add_tool(Database(conn), "db")
agent.add_tool(Charts(), "charts")
agent.add_tool(Documents(), "docs")
agent.add_function(answer)

# Defer everything except answer and the DB query tool
agent.enable_tool_search(
    always_loaded=["answer", "Database-db__query"],
)
```

**Parameters:**

- `always_loaded`: Tool names to keep always visible to the LLM. If `None`, only the search tool is visible.
- `search_fn`: Custom async search function with signature `async def search(query: str, catalog: list[dict]) -> list[str]`. Each catalog entry has `name`, `description`, and `args` keys. If `None`, uses built-in keyword matching.
- `search_model`: Model for LLM-based search (only used with custom `search_fn`). Defaults based on provider.

**How it works:**

1. Deferred tools are sent to the API with `defer_loading: true` — the provider hides them from the LLM's context window
2. When the LLM needs a tool, it calls `tool_search(query="...")`
3. The search function matches the query against tool names, descriptions, and argument names
4. Matching tools are returned as `tool_reference` blocks, which the API auto-expands into full definitions
5. The LLM can now call the discovered tools

**Behavior with `__llm__()`:** When tool search is enabled, `__llm__()` is only called for tools that have at least one always-loaded method. Fully deferred tools skip their `__llm__()` output to save context tokens.

**Categories hint:** A short summary of deferred tool categories is automatically injected into the system prompt so the LLM knows what kinds of tools are available to search for.

### Sub-Agent Management

#### add_sub_agent(agent, name=None, description=None, compute_levels=None) -> str

Register a sub-agent that can be triggered by this agent. The sub-agent runs its own conversation loop and returns only its final response.

```python
researcher = Agentlys(name="researcher", instruction="You research topics.", provider="anthropic")
parent.add_sub_agent(researcher)
# Returns: "sub_agent__researcher"

# With dynamic compute levels (parent LLM picks high/medium/low per call)
parent.add_sub_agent(researcher, compute_levels=True)
```

#### remove_sub_agent(name)

Remove a sub-agent by name (without the `sub_agent__` prefix).

```python
parent.remove_sub_agent("researcher")
```

#### on_sub_agent_event

Optional callback for observing sub-agent events. Signature: `(name: str, invocation_id: str, event: dict) -> None`.

```python
parent.on_sub_agent_event = lambda name, id, event: print(f"[{name}] {event}")
```

See [Sub-Agents](sub-agents.md) for full documentation.

### Template Methods

#### from_template(file_path) -> Agentlys

Create an agent from a template file.

```python
agent = Agentlys.from_template("templates/coding_assistant.md")
```

## Message Class

### Constructor

```python
Message(
    role: Literal["user", "assistant", "function"],
    content: str = None,
    name: str = None,
    function_call: dict = None,
    function_call_id: str = None,
    image: PILImage.Image = None,
    parts: list[MessagePart] = None
)
```

### Methods

#### to_markdown() -> str

Convert message to markdown format for display.

#### to_terminal(display_image=False) -> str

Convert message to terminal-friendly format.

#### to_dict() -> dict

Convert message to dictionary format.

## MessagePart Class

Represents a part of a message (text, image, function call, etc.).

```python
MessagePart(
    type: Literal["text", "image", "function_call", "function_result", "function_result_image"],
    content: str = None,
    image: PILImage.Image = None,
    function_call: dict = None,
    function_call_id: str = None
)
```

## APIProvider Enum

Available LLM providers:

```python
from agentlys import APIProvider

APIProvider.OPENAI              # OpenAI GPT models
APIProvider.ANTHROPIC           # Anthropic Claude models
APIProvider.OPENAI_FUNCTION_LEGACY  # Legacy OpenAI function calling
```

## Environment Variables

- `AGENTLYS_MODEL`: Default model to use
- `AGENTLYS_HOST`: Custom provider endpoint
- `AGENTLYS_OUTPUT_SIZE_LIMIT`: Maximum output size in characters (default: 4000)
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key

## Error Handling

### StopLoopException

Raised to stop conversation loops.

```python
from agentlys.chat import StopLoopException

try:
    for message in agent.run_conversation("Hello"):
        print(message.content)
except StopLoopException:
    print("Conversation ended")
```

## Custom Providers

Create custom providers by extending `BaseProvider`:

```python
from agentlys.providers.base_provider import BaseProvider

class MyCustomProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_completion(self, messages, **kwargs):
        # Implement your provider logic
        pass

agent = Agentlys(provider=MyCustomProvider)
```
