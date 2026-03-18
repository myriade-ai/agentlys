# Sub-Agents

Delegate tasks to specialized child agents. Each sub-agent runs its own conversation loop and only returns its final response — keeping the parent's context window clean.

```python
from agentlys import Agentlys

researcher = Agentlys(
    name="researcher",
    instruction="You are a research specialist. Find and synthesize information.",
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

writer = Agentlys(
    name="writer",
    instruction="You are an expert technical writer.",
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

coordinator = Agentlys(
    instruction="You coordinate research and writing tasks. Use your sub-agents to delegate work.",
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

coordinator.add_sub_agent(researcher)
coordinator.add_sub_agent(writer)

for message in coordinator.run_conversation("Write a report on AI agent architectures"):
    print(message.content)
```

The coordinator's LLM sees `sub_agent__researcher` and `sub_agent__writer` as callable tools. It decides when and how to delegate.

---

## How It Works

1. You register an `Agentlys` instance as a sub-agent on a parent agent
2. A tool named `sub_agent__<name>` is created with a single `prompt` parameter
3. When the parent LLM calls this tool, the sub-agent:
   - Resets its message history (stateless per invocation)
   - Runs a full conversation loop with its own tools
   - Returns only its final assistant response to the parent
4. Multiple sub-agent calls in the same turn execute **in parallel** via `asyncio.gather()`

---

## API

### add_sub_agent(agent, name=None, description=None, compute_levels=None) -> str

Register a sub-agent.

| Parameter        | Type                 | Default             | Description                              |
| ---------------- | -------------------- | ------------------- | ---------------------------------------- |
| `agent`          | `Agentlys`           | required            | The agent instance to register           |
| `name`           | `str`                | `agent.name`        | Override the tool name                   |
| `description`    | `str`                | `agent.instruction` | Description shown to the parent LLM      |
| `compute_levels` | `bool` or `dict`     | `None`              | Enable dynamic compute levels (see below)|

Returns the registered function name (e.g. `"sub_agent__researcher"`).

```python
# Use defaults
coordinator.add_sub_agent(researcher)

# Override name and description
coordinator.add_sub_agent(
    researcher,
    name="topic_explorer",
    description="Deep dive research on any topic. Returns a structured summary."
)
```

Raises `ValueError` if the agent has no name or a sub-agent with that name already exists.

### remove_sub_agent(name)

Remove a sub-agent by name (without the `sub_agent__` prefix).

```python
coordinator.remove_sub_agent("researcher")
```

### on_sub_agent_event

Optional callback to observe sub-agent execution events (streaming text, tool calls, etc.). Called with `(sub_agent_name, invocation_id, event)` for each event during a sub-agent's conversation.

```python
def on_event(name: str, invocation_id: str, event: dict):
    if event["type"] == "text":
        print(f"[{name}] {event['content']}", end="", flush=True)

coordinator.on_sub_agent_event = on_event
```

Set to `None` to disable (default). Cleared by `reset()`.

---

## Sub-Agents with Their Own Tools

Sub-agents are full `Agentlys` instances — they can have tools, functions, MCP servers, and even their own sub-agents.

```python
from agentlys import Agentlys

def search_web(query: str) -> str:
    """Search the web for information"""
    return fetch_results(query)

def read_url(url: str) -> str:
    """Read the contents of a URL"""
    return fetch_page(url)

researcher = Agentlys(
    name="researcher",
    instruction="You research topics using web search and URL reading.",
    provider="anthropic"
)
researcher.add_function(search_web)
researcher.add_function(read_url)

coordinator = Agentlys(
    instruction="Delegate research tasks to your sub-agent.",
    provider="anthropic"
)
coordinator.add_sub_agent(researcher)
```

When the coordinator calls the researcher, the researcher can make multiple tool calls (search, read URLs, search again) before returning its final answer.

---

## Mixed Providers and Models

Each sub-agent can use a different provider and model:

```python
fast_agent = Agentlys(
    name="quick_lookup",
    instruction="Answer simple factual questions.",
    provider="openai",
    model="gpt-4o-mini"
)

deep_agent = Agentlys(
    name="deep_analysis",
    instruction="Perform thorough analysis.",
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

coordinator = Agentlys(provider="anthropic")
coordinator.add_sub_agent(fast_agent, description="Quick factual lookups")
coordinator.add_sub_agent(deep_agent, description="In-depth analysis tasks")
```

---

## Nested Sub-Agents

Sub-agents can have their own sub-agents:

```python
analyst = Agentlys(name="analyst", instruction="Analyze data", provider="anthropic")
visualizer = Agentlys(name="visualizer", instruction="Create charts", provider="anthropic")

data_team = Agentlys(name="data_team", instruction="Coordinate data analysis and visualization", provider="anthropic")
data_team.add_sub_agent(analyst)
data_team.add_sub_agent(visualizer)

coordinator = Agentlys(instruction="You manage a data team.", provider="anthropic")
coordinator.add_sub_agent(data_team)
```

---

## Async and Streaming

Sub-agents work with both async and streaming modes:

```python
# Async
async for message in coordinator.run_conversation_async("Analyze the dataset"):
    print(message.content)

# Streaming
async for event in coordinator.run_conversation_stream_async("Analyze the dataset"):
    if event["type"] == "text":
        print(event["content"], end="", flush=True)
```

---

## Compute Levels

Let the parent agent dynamically choose how much compute to allocate per sub-agent call. Simple lookups use a lightweight model, complex reasoning uses a powerful one.

```python
from agentlys import Agentlys

researcher = Agentlys(
    name="researcher",
    instruction="You research topics thoroughly.",
    provider="anthropic"
)

coordinator = Agentlys(
    instruction="Delegate tasks. Use 'high' compute for complex analysis, 'low' for simple lookups.",
    provider="anthropic"
)

# Enable with default mapping (Opus / Sonnet / Haiku)
coordinator.add_sub_agent(researcher, compute_levels=True)
```

The parent LLM now sees an optional `compute_level` parameter (`"high"`, `"medium"`, `"low"`) on the sub-agent tool. It defaults to `"medium"` if omitted.

### Default model mapping

| Level    | Model                        |
| -------- | ---------------------------- |
| `high`   | `claude-opus-4-20250514`     |
| `medium` | `claude-sonnet-4-20250514`   |
| `low`    | `claude-haiku-4-5-20251001`  |

### Custom model mapping

Override the defaults for any provider:

```python
coordinator.add_sub_agent(researcher, compute_levels={
    "high": "gpt-4o",
    "medium": "gpt-4o-mini",
    "low": "gpt-3.5-turbo",
})
```

All three keys (`"high"`, `"medium"`, `"low"`) are required.

### Without compute levels

When `compute_levels` is not set (default), the sub-agent always runs on whatever model it was configured with — no `compute_level` parameter is exposed to the parent LLM.

---

## Key Behaviors

- **Stateless**: Sub-agent messages are cleared before each invocation — no memory between calls
- **Context-lean**: Only the final response reaches the parent, not the sub-agent's full conversation
- **Parallel**: Multiple sub-agents called in the same turn run concurrently
- **Isolated**: Sub-agent state (tools, messages, config) is independent from the parent
