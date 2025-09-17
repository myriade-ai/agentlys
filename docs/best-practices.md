# Best Practices for Building AI Agents

This guide covers best practices for building robust, reliable AI agents with Agentlys.

## Agent Design Principles

### 1. Clear Instructions

Write specific, actionable instructions:

```python
# ❌ Vague instruction
agent = Agentlys(instruction="Help with coding")

# ✅ Clear, specific instruction
agent = Agentlys(instruction="""
You are a Python expert who helps with:
1. Code review and optimization
2. Debugging and error resolution
3. Writing clean, maintainable code
4. Following PEP 8 standards

Always provide working code examples and explain your reasoning.
""")
```

### 2. Structured Workflows

Design agents with clear workflows:

```python
DEVELOPER_INSTRUCTION = """
### Role
You are a senior developer agent with access to development tools.

### Workflow
1. Understand the requirements thoroughly
2. Plan the implementation approach
3. Set up the development environment
4. Implement the solution incrementally
5. Test thoroughly at each step
6. Refactor and optimize
7. Document the solution
8. Create pull requests with clear descriptions

### Tools Usage
- Use CodeEditor for all file operations
- Use Terminal for running commands and tests
- Use Git for version control operations
- Always run tests before committing

### Quality Standards
- Write clean, readable code
- Include comprehensive error handling
- Add appropriate documentation
- Follow established patterns in the codebase
"""

agent = Agentlys(
    instruction=DEVELOPER_INSTRUCTION,
    provider="anthropic",  # Recommended for complex workflows
    model="claude-sonnet-4-20250514"
)
```

## Tool Design Best Practices

### 1. Granular Tool Functions

Break complex operations into smaller, focused functions:

```python
# ❌ Monolithic function
def manage_database(action: str, table: str, data: dict = None, query: str = None):
    """Do everything with database"""
    # Complex logic handling multiple concerns
    pass

# ✅ Granular functions
class DatabaseManager:
    def create_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create a new table with the specified schema"""
        pass

    def insert_record(self, table_name: str, data: Dict) -> str:
        """Insert a single record and return the ID"""
        pass

    def query_records(self, table_name: str, conditions: Dict = None, limit: int = 100) -> List[Dict]:
        """Query records with optional conditions"""
        pass

    def update_record(self, table_name: str, record_id: str, data: Dict) -> bool:
        """Update a specific record"""
        pass
```

### 2. Comprehensive Error Handling

Always handle errors gracefully and provide actionable feedback:

```python
def read_config_file(file_path: str) -> Dict[str, any]:
    """Read and parse a configuration file"""
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"Configuration file not found: {file_path}",
                "suggestion": "Create the config file or check the path"
            }

        with open(file_path, 'r') as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ["api_key", "base_url"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            return {
                "success": False,
                "error": f"Missing required fields: {missing_fields}",
                "suggestion": f"Add these fields to {file_path}"
            }

        return {
            "success": True,
            "config": config
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON in config file: {str(e)}",
            "suggestion": "Check JSON syntax in the config file"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error reading config: {str(e)}",
            "suggestion": "Check file permissions and try again"
        }
```

### 3. Rich Return Values

Provide detailed, structured responses:

```python
def run_tests(test_path: str = "tests/") -> Dict[str, any]:
    """Run tests and return detailed results"""
    start_time = time.time()

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300
        )

        duration = time.time() - start_time

        return {
            "success": result.returncode == 0,
            "duration": round(duration, 2),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": {
                "passed": result.stdout.count(" PASSED"),
                "failed": result.stdout.count(" FAILED"),
                "errors": result.stdout.count(" ERROR"),
                "skipped": result.stdout.count(" SKIPPED")
            },
            "recommendation": "All tests passed! 🎉" if result.returncode == 0 else "Some tests failed. Check the output above."
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Tests timed out after 5 minutes",
            "suggestion": "Check for hanging tests or infinite loops"
        }
```

## Conversation Management

### 1. Context Management

Keep conversations focused and manage context effectively:

```python
class ContextManagedAgent:
    def __init__(self):
        self.agent = Agentlys(
            instruction="You are a focused assistant",
            max_interactions=20  # Prevent runaway conversations
        )
        self.conversation_history = []

    def ask_with_context(self, query: str, include_history: bool = True):
        if include_history and self.conversation_history:
            # Include relevant history
            context = f"Previous context:\n{self.get_relevant_history()}\n\nNew query: {query}"
        else:
            context = query

        response = self.agent.ask(context)

        # Store in history
        self.conversation_history.append({
            "query": query,
            "response": response.content,
            "timestamp": datetime.now()
        })

        return response

    def get_relevant_history(self) -> str:
        # Return last few relevant interactions
        return "\n".join([
            f"Q: {item['query']}\nA: {item['response'][:200]}..."
            for item in self.conversation_history[-3:]
        ])
```

### 2. Conversation Stopping Conditions

Implement smart stopping conditions:

```python
class SmartAgent(Agentlys):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_complete_indicators = [
            "task completed",
            "implementation finished",
            "all tests passing",
            "pull request created"
        ]

    def custom_stopping_condition(self, message: Message) -> bool:
        """Custom logic to determine when to stop the conversation"""
        content = message.content.lower()

        # Stop if task completion indicators are present
        if any(indicator in content for indicator in self.task_complete_indicators):
            return True

        # Stop if agent is asking for clarification repeatedly
        if content.count("clarification") > 2:
            return True

        return False

    def run_conversation(self, prompt: str):
        for message in super().run_conversation(prompt):
            yield message

            if self.custom_stopping_condition(message):
                break
```

## Performance Optimization

### 1. Efficient Tool Calls

Design tools to minimize back-and-forth:

```python
# ❌ Multiple separate calls
def get_file_info(filename: str) -> str:
    """Get basic file info"""
    return f"Size: {os.path.getsize(filename)} bytes"

def get_file_content(filename: str) -> str:
    """Get file content"""
    with open(filename) as f:
        return f.read()

# ✅ Combined efficient call
def analyze_file(filename: str, include_content: bool = True, max_size: int = 10000) -> Dict:
    """Analyze a file and return comprehensive information"""
    try:
        stat = os.stat(filename)
        result = {
            "filename": filename,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "readable": os.access(filename, os.R_OK),
            "writable": os.access(filename, os.W_OK)
        }

        if include_content and stat.st_size <= max_size:
            with open(filename, 'r') as f:
                result["content"] = f.read()
                result["lines"] = len(result["content"].splitlines())
        elif stat.st_size > max_size:
            result["content_note"] = f"File too large ({stat.st_size} bytes). Set max_size higher to include content."

        return result

    except Exception as e:
        return {"error": str(e)}
```

### 2. Caching and Memoization

Cache expensive operations:

```python
from functools import lru_cache
import hashlib

class CachedTools:
    def __init__(self):
        self._cache = {}

    def expensive_analysis(self, data: str) -> Dict:
        """Perform expensive analysis with caching"""
        # Create cache key
        cache_key = hashlib.md5(data.encode()).hexdigest()

        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            cached_result["cached"] = True
            return cached_result

        # Perform expensive operation
        result = self._do_expensive_analysis(data)
        result["cached"] = False

        # Cache the result
        self._cache[cache_key] = result

        return result

    @lru_cache(maxsize=128)
    def get_api_data(self, endpoint: str) -> Dict:
        """Cache API responses"""
        # API call implementation
        pass
```

## Error Recovery and Resilience

### 1. Graceful Degradation

Implement fallback strategies:

```python
class ResilientFileManager:
    def __init__(self):
        self.backup_locations = ["./backup/", "/tmp/backup/", "./"]

    def save_file(self, filename: str, content: str) -> Dict:
        """Save file with fallback locations"""
        for location in self.backup_locations:
            try:
                os.makedirs(location, exist_ok=True)
                filepath = os.path.join(location, filename)

                with open(filepath, 'w') as f:
                    f.write(content)

                return {
                    "success": True,
                    "location": filepath,
                    "message": f"File saved to {filepath}"
                }

            except Exception as e:
                continue

        return {
            "success": False,
            "error": "Could not save file to any location",
            "attempted_locations": self.backup_locations
        }
```

### 2. Retry Logic

Implement smart retry mechanisms:

```python
import time
from typing import Callable, Any

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_multiplier: float = 2.0
) -> Any:
    """Retry a function with exponential backoff"""

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e

            delay = base_delay * (backoff_multiplier ** attempt)
            time.sleep(delay)

def network_request(url: str) -> Dict:
    """Make a network request with retry logic"""
    def _make_request():
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    try:
        return {
            "success": True,
            "data": retry_with_backoff(_make_request)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Check network connection and URL"
        }
```

## Security Best Practices

### 1. Input Validation

Always validate and sanitize inputs:

```python
import os
import re
from pathlib import Path

def safe_file_operation(filename: str, operation: str = "read") -> Dict:
    """Safely perform file operations with validation"""

    # Validate filename
    if not filename or ".." in filename or filename.startswith("/"):
        return {
            "success": False,
            "error": "Invalid filename: path traversal not allowed"
        }

    # Validate against allowed patterns
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        return {
            "success": False,
            "error": "Filename contains invalid characters"
        }

    # Ensure file is within allowed directory
    allowed_dir = Path("./workspace").resolve()
    target_path = (allowed_dir / filename).resolve()

    if not str(target_path).startswith(str(allowed_dir)):
        return {
            "success": False,
            "error": "File access outside allowed directory"
        }

    # Proceed with operation
    try:
        if operation == "read":
            with open(target_path, 'r') as f:
                return {
                    "success": True,
                    "content": f.read()
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### 2. Secret Management

Handle secrets securely:

```python
import os
from typing import Optional

class SecretManager:
    def __init__(self):
        self._secrets = {}

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment or secure storage"""
        # Try environment first
        if key in os.environ:
            return os.environ[key]

        # Try local secure storage
        if key in self._secrets:
            return self._secrets[key]

        return None

    def set_secret(self, key: str, value: str) -> bool:
        """Set secret (in memory only, not persisted)"""
        self._secrets[key] = value
        return True

    def api_call_with_auth(self, url: str, secret_key: str = "API_KEY") -> Dict:
        """Make authenticated API call"""
        api_key = self.get_secret(secret_key)

        if not api_key:
            return {
                "success": False,
                "error": f"API key '{secret_key}' not found in environment or secrets"
            }

        # Make API call (don't log the key!)
        headers = {"Authorization": f"Bearer {api_key}"}
        # ... rest of implementation
```

## Testing and Validation

### 1. Agent Testing

Test your agents systematically:

```python
import pytest
from unittest.mock import Mock, patch

class TestDeveloperAgent:
    def setup_method(self):
        self.agent = Agentlys(
            instruction="You are a test developer agent",
            provider="openai",  # Use cheaper model for testing
            model="gpt-5-mini"
        )

        # Add mock tools for testing
        self.mock_editor = Mock()
        self.agent.add_tool(self.mock_editor, "Editor")

    def test_simple_coding_task(self):
        """Test agent handles simple coding tasks"""
        response = self.agent.ask("Create a function to add two numbers")

        assert "def" in response.content
        assert "add" in response.content.lower()
        assert response.role == "assistant"

    @patch('subprocess.run')
    def test_tool_interaction(self, mock_subprocess):
        """Test agent interacts with tools correctly"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Tests passed"

        terminal = Terminal()
        self.agent.add_tool(terminal)

        responses = list(self.agent.run_conversation("Run the test suite"))

        # Verify tool was called
        assert any("run_command" in str(response) for response in responses)

    def test_error_handling(self):
        """Test agent handles errors gracefully"""
        # Simulate tool error
        self.mock_editor.read_file.side_effect = FileNotFoundError("File not found")

        response = self.agent.ask("Read the file config.json")

        # Agent should handle the error gracefully
        assert "error" in response.content.lower() or "not found" in response.content.lower()
```

### 2. Tool Testing

Test tools independently:

```python
def test_file_manager_tools():
    """Test file manager tools work correctly"""
    fm = FileManager(temp_dir)

    # Test file creation
    result = fm.write_file("test.txt", "Hello, World!")
    assert result["success"] == True

    # Test file reading
    result = fm.read_file("test.txt")
    assert result["success"] == True
    assert result["content"] == "Hello, World!"

    # Test error cases
    result = fm.read_file("nonexistent.txt")
    assert result["success"] == False
    assert "not found" in result["error"].lower()
```

By following these best practices, you'll build more reliable, maintainable, and effective AI agents that can handle complex tasks while remaining robust and secure.
