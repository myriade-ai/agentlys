# Agentlys

[![image](https://img.shields.io/pypi/v/agentlys.svg)](https://pypi.python.org/pypi/agentlys)
[![image](https://img.shields.io/github/license/myriade-ai/agentlys)](https://github.com/myriade-ai/agentlys/blob/master/LICENSE)
[![Actions status](https://github.com/myriade-ai/agentlys/actions/workflows/test.yml/badge.svg)](https://github.com/myriade-ai/agentlys/actions)

**Turn any Python class into an AI tool. Instantly.**

Async-native • MCP support • Multi-providers • ~500 lines of core code

```python
class Database:
    def __llm__(self):
        return f"Tables: {self.list_tables()}"  # AI sees this every turn

    def query(self, sql: str) -> list[dict]:
        """Execute SQL query"""
        return self.execute(sql)

    def describe(self, table: str) -> dict:
        """Get table schema"""
        return self.get_schema(table)

agent = Agentlys()
agent.add_tool(Database(conn))  # That's it. All methods are now AI tools.
agent.run("What drove revenue decline in Q3?")
```

**Other frameworks**: 50 lines of tool definitions, separate schemas, manual state management.  
**Agentlys**: Your class IS the tool. Methods become actions. `__llm__()` injects state.

---

## Why Agentlys?

| If you want...                        | Use          |
| ------------------------------------- | ------------ |
| Graphs and state machines             | LangGraph    |
| Team-based agent crews                | CrewAI       |
| **Your existing classes as AI tools** | **Agentlys** |

**~500 lines of core code.** No framework lock-in. No magic.

---

## Install

```bash
pip install 'agentlys[all]'  # OpenAI + Anthropic + MCP
```

---

## The Pattern

### 1. Functions → Tools

```python
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return requests.get(f"https://wttr.in/{city}?format=3").text

agent.add_function(get_weather)
```

### 2. Classes → Stateful Tools (the killer feature)

```python
class FileSystem:
    def __init__(self, root: str):
        self.root = root

    def __llm__(self):
        """State shown to AI each turn"""
        return f"Current directory: {self.root}\nFiles: {os.listdir(self.root)}"

    def read(self, path: str) -> str:
        """Read file contents"""
        return open(f"{self.root}/{path}").read()

    def write(self, path: str, content: str):
        """Write to file"""
        open(f"{self.root}/{path}", 'w').write(content)

agent.add_tool(FileSystem("/workspace"))
# AI now sees file state, can read/write, all from one class
```

### 3. Run Conversations

```python
for message in agent.run_conversation("Refactor config.json to use environment variables"):
    print(message.content)
```

## Async Support

```python
# Async conversation loop
async for message in agent.run_conversation_async("Analyze the data"):
    print(message.content)

# Single async call
response = await agent.ask_async("What tables exist?")
```

---

## Real Example: [agentlys-dev](https://github.com/myriade-ai/agentlys-dev)

A coding agent in 15 lines:

```python
from agentlys import Agentlys
from agentlys_tools import CodeEditor, Terminal, Git

agent = Agentlys(
    instruction="You are a senior developer",
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)

agent.add_tool(CodeEditor())
agent.add_tool(Terminal())
agent.add_tool(Git())

agent.run_conversation("Create a FastAPI app with tests")
```

---

## Providers

```python
# Anthropic (default)
agent = Agentlys(provider="anthropic", model="claude-sonnet-4-20250514")

# OpenAI
agent = Agentlys(model="gpt-4o")
```

---

## More

- [API Reference](docs/api-reference.md)
- [Examples](examples/)
- [MCP Integration](docs/mcp.md)

---

## Used By

- [Myriade](https://www.myriade.ai) — AI-native data platform

## When NOT to use Agentlys

- You need graph-based workflows → Use LangGraph
- You want pre-built agent teams → Use CrewAI
- You need sandboxed code execution → Use Smolagents

Agentlys is for: **turning your existing Python code into AI tools with zero ceremony.**

## License

MIT
