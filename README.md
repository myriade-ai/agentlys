# Agentlys

[![image](https://img.shields.io/pypi/v/agentlys.svg)](https://pypi.python.org/pypi/agentlys)
[![image](https://img.shields.io/github/license/myriade-ai/agentlys)](https://github.com/myriade-ai/agentlys/blob/master/LICENSE)
[![Actions status](https://github.com/myriade-ai/agentlys/actions/workflows/test.yml/badge.svg)](https://github.com/myriade-ai/agentlys/actions)

> ⚠️ **Warning**: Since agentic capabilities are evolving fast, expect the API to change.

A lightweight Python library for building AI agents. Turn any Python function or class into AI tools. Supports OpenAI, Anthropic, async operations, and streaming conversations.

## Features

- Add functions: `agent.add_function(my_function)`
- Add classes: `agent.add_tool(my_class_instance)`
- MCP support
- Multiple providers: OpenAI, Anthropic, ...
- Async/await support
- Image processing
- Conversation streaming
- Template system

## Real-World Example: Code Development Agent

Agentlys excels at building development agents. Here's how [agentlys-dev](https://github.com/myriade-ai/agentlys-dev) uses agentlys to create an AI developer:

```python
from agentlys import Agentlys
# pip install 'agentlys-tools[all]'
from agentlys_tools.code_editor import CodeEditor
from agentlys_tools.terminal import Terminal

# Create a developer agent
agent = Agentlys(
    instruction="""You are a developer agent equipped with tools to:
    1. Edit code files
    2. Run terminal commands
    3. Test and debug applications""",
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    name="Developer"
)

# Add development tools
code_editor = CodeEditor()
agent.add_tool(code_editor)

terminal = Terminal()
agent.add_tool(terminal)

# The agent can now autonomously develop, test, and deploy code
for message in agent.run_conversation("Create a FastAPI hello world app with tests"):
    print(message.to_markdown())
```

## Installation

Install agentlys with all providers and features:

```bash
pip install 'agentlys[all]'
```

Or install with specific providers:

```bash
# OpenAI only
pip install 'agentlys[openai]'

# Anthropic only
pip install 'agentlys[anthropic]'

# With MCP support (Python 3.10+)
pip install 'agentlys[mcp]'
```

## Usage

### Functions

Turn regular Python functions into tools by using `add_function()`

- Methods docstring, args and return type will be used to generate the tool description.

```python
from agentlys import Agentlys

def get_weather(city: str) -> str:
    return f"Sunny in {city}"

agent = Agentlys()
agent.add_function(get_weather)
agent.ask("What's the weather in Tokyo?")
```

### Classes (the killer feature)

Turn entire classes into tools by using `add_tool()`

- Methods docstring, args and return type will be used to generate the tool description.
- \_\_llm\_\_ method will be used to give AI the last state of the tool at each interaction.

```python
import os

class FileManager:
    def __llm__(self):
        return "Files:\n" + "\n".join(os.listdir(self.directory))

    def read_file(self, path: str) -> str:
        """Read a file
        Args:
            path: Path is relative to the directory or absolute
        """
        with open(path) as f:
            return f.read()

    def write_file(self, path: str, content: str):
        with open(path, 'w') as f:
            f.write(content)

file_manager = FileManager()
agent = Agentlys()
agent.add_tool(file_manager)

# AI can now read/write files
for msg in agent.run_conversation("Read config.json and update the port to 8080"):
    print(msg.content)
```

## Advanced Features

### Image Support

```python
from agentlys import Agentlys, Message
from PIL import Image

agent = Agentlys()
image = Image.open("examples/image.jpg")
message = Message(role="user", content="Describe this image", image=image)
response = agent.ask(message)
```

### Template System

```python
# Load agent from markdown template
agent = Agentlys.from_template("./agent_template.md")
```

### Async Support

```python
# Async operations
response = await agent.ask_async("Hello")
async for message in agent.run_conversation_async("Help me code"):
    print(message.content)
```

## Configuration

```bash
# Set up your API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Choose your model (optional)
export AGENTLYS_MODEL="claude-sonnet-4-20250514"  # or gpt-5-mini
```

💡 **Recommendation**: Use Anthropic's Claude models for complex agentic behavior and tool use.

## Use Cases

- **🤖 AI Assistants**: Build conversational assistants with tool access
- **🛠️ Development Agents**: Create agents that can code, test, and deploy (like [agentlys-dev](https://github.com/myriade-ai/agentlys-dev))
- **📊 Data Analysis**: Agents that can query databases, generate reports, visualize data
- **🌐 Web Automation**: Agents that interact with web APIs and services
- **📋 Task Automation**: Automate complex workflows with AI decision-making
- **🎯 Custom Tools**: Integrate your existing Python tools with AI

## Documentation

- [API Reference](docs/api-reference.md) - Complete API documentation
- [Examples](examples/) - More example implementations
- [Provider Guide](docs/providers.md) - Working with different LLM providers
- [Tool Development](docs/tool-development.md) - Creating custom tools
- [Best Practices](docs/best-practices.md) - Tips for building robust agents

## Support

If you encounter any issues or have questions, please file an issue on the GitHub project page.

## License

This project is licensed under the terms of the MIT license.
