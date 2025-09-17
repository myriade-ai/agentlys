# Tool Development Guide

Tools are the key to building powerful AI agents with Agentlys. This guide covers how to create effective tools that your AI agents can use.

## Function-Based Tools

### Basic Function Tools

The simplest way to add tools is by decorating regular Python functions:

```python
from agentlys import Agentlys

def get_current_time() -> str:
    """Get the current time in ISO format"""
    from datetime import datetime
    return datetime.now().isoformat()

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> float:
    """Calculate tip amount for a bill

    Args:
        bill_amount: The total bill amount
        tip_percentage: Tip percentage (default 15%)

    Returns:
        The tip amount
    """
    return bill_amount * (tip_percentage / 100)

agent = Agentlys()
agent.add_function(get_current_time)
agent.add_function(calculate_tip)
```

### Type Hints Are Critical

Always use proper type hints - they're converted to JSON schema for the LLM:

```python
from typing import List, Dict, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

def create_task(
    title: str,
    description: Optional[str] = None,
    priority: Priority = Priority.MEDIUM,
    tags: List[str] = None,
    metadata: Dict[str, str] = None
) -> Dict[str, any]:
    """Create a new task with the given parameters"""
    return {
        "title": title,
        "description": description or "",
        "priority": priority.value,
        "tags": tags or [],
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat()
    }
```

### Async Function Tools

Agentlys supports async functions seamlessly:

```python
import asyncio
import aiohttp

async def fetch_url(url: str) -> str:
    """Fetch content from a URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def process_data(data: List[str]) -> List[str]:
    """Process a list of data asynchronously"""
    await asyncio.sleep(0.1)  # Simulate async work
    return [item.upper() for item in data]

agent = Agentlys()
agent.add_function(fetch_url)
agent.add_function(process_data)
```

## Class-Based Tools

### Basic Class Tools

Turn entire classes into tools by using `add_tool()`:

```python
class FileManager:
    def __init__(self, base_path: str = "."):
        self.base_path = base_path

    def read_file(self, filename: str) -> str:
        """Read contents of a file"""
        with open(os.path.join(self.base_path, filename), 'r') as f:
            return f.read()

    def write_file(self, filename: str, content: str) -> bool:
        """Write content to a file"""
        try:
            with open(os.path.join(self.base_path, filename), 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            return False

    def list_files(self) -> List[str]:
        """List all files in the directory"""
        return os.listdir(self.base_path)

file_manager = FileManager("/tmp")
agent = Agentlys()
agent.add_tool(file_manager, "FileManager")
```

### Advanced Class Tools with Context

Classes can maintain state and context between tool calls:

```python
class DatabaseConnection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.transaction_count = 0

    def __llm__(self) -> str:
        """This method provides context to the LLM about the tool's state"""
        status = "connected" if self.connection else "disconnected"
        return f"Database status: {status}, Transactions: {self.transaction_count}"

    def connect(self) -> bool:
        """Connect to the database"""
        try:
            import sqlite3
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception:
            return False

    def execute_query(self, query: str) -> List[Dict]:
        """Execute a SQL query and return results"""
        if not self.connection:
            raise ValueError("Not connected to database")

        cursor = self.connection.cursor()
        cursor.execute(query)
        self.transaction_count += 1

        # Convert to list of dicts
        columns = [description[0] for description in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))

        return results

    def close(self) -> bool:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            return True
        return False

db = DatabaseConnection("data.db")
agent = Agentlys()
agent.add_tool(db, "Database")
```

## Tool Design Best Practices

### 1. Clear Documentation

Always provide detailed docstrings:

```python
def search_documents(
    query: str,
    limit: int = 10,
    include_content: bool = False
) -> List[Dict[str, str]]:
    """Search through documents using text similarity

    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        include_content: Whether to include full document content (default: False)

    Returns:
        List of document matches with metadata

    Examples:
        search_documents("python tutorial", limit=5)
        search_documents("machine learning", include_content=True)
    """
    # Implementation here
```

### 2. Error Handling

Handle errors gracefully and provide meaningful messages:

```python
def send_email(to: str, subject: str, body: str) -> Dict[str, any]:
    """Send an email to the specified recipient"""
    try:
        # Email sending logic
        return {
            "success": True,
            "message": f"Email sent to {to}",
            "message_id": "12345"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Check your email configuration and network connection"
        }
```

### 3. Input Validation

Validate inputs and provide helpful error messages:

```python
def resize_image(width: int, height: int, image_path: str) -> str:
    """Resize an image to the specified dimensions"""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        raise ValueError("Unsupported image format")

    # Resize logic here
    return f"Image resized to {width}x{height}"
```

### 4. Chunked Output for Large Results

Handle large outputs appropriately:

```python
def analyze_large_dataset(file_path: str, chunk_size: int = 1000) -> Iterator[Dict]:
    """Analyze a large dataset in chunks to avoid memory issues"""
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield {"chunk_results": process_chunk(chunk)}
                chunk = []

        if chunk:  # Process remaining items
            yield {"final_results": process_chunk(chunk)}
```

## Tool Integration Patterns

### 1. Tool Chaining

Design tools that work well together:

```python
class WebScraper:
    def fetch_page(self, url: str) -> str:
        """Fetch HTML content from a URL"""
        # Implementation
        pass

    def extract_links(self, html: str) -> List[str]:
        """Extract all links from HTML content"""
        # Implementation
        pass

    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML"""
        # Implementation
        pass

# Agent can now: fetch_page() -> extract_links() -> fetch_page() for each link
```

### 2. Stateful Tools

Tools that maintain state across calls:

```python
class TaskManager:
    def __init__(self):
        self.tasks = []
        self.completed_tasks = []

    def add_task(self, task: str, priority: int = 1) -> str:
        """Add a new task"""
        task_id = f"task_{len(self.tasks) + 1}"
        self.tasks.append({
            "id": task_id,
            "description": task,
            "priority": priority,
            "created_at": datetime.now()
        })
        return f"Added task {task_id}"

    def complete_task(self, task_id: str) -> str:
        """Mark a task as completed"""
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if task:
            self.tasks.remove(task)
            self.completed_tasks.append({**task, "completed_at": datetime.now()})
            return f"Task {task_id} completed"
        return f"Task {task_id} not found"

    def list_tasks(self) -> List[Dict]:
        """List all pending tasks"""
        return self.tasks
```

### 3. Configuration Tools

Tools that can be configured by the agent:

```python
class APIClient:
    def __init__(self):
        self.base_url = None
        self.headers = {}
        self.timeout = 30

    def configure(self, base_url: str, headers: Dict[str, str] = None, timeout: int = 30):
        """Configure the API client"""
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        return "API client configured"

    def get(self, endpoint: str) -> Dict:
        """Make a GET request to the API"""
        if not self.base_url:
            raise ValueError("API client not configured. Call configure() first.")
        # Make request
        pass
```

## Testing Tools

Always test your tools independently:

```python
def test_calculator_tool():
    calc = Calculator()

    # Test basic functionality
    assert calc.add(2, 3) == 5
    assert calc.multiply(4, 5) == 20

    # Test edge cases
    assert calc.divide(10, 0) == {"error": "Cannot divide by zero"}

    # Test with agent
    agent = Agentlys()
    agent.add_tool(calc)

    response = agent.ask("What is 15 + 25?")
    assert "40" in response.content

if __name__ == "__main__":
    test_calculator_tool()
```

## Performance Considerations

1. **Avoid Heavy Computations**: Keep tool operations fast
2. **Use Async When Appropriate**: For I/O operations
3. **Implement Caching**: For expensive operations
4. **Limit Output Size**: Large outputs can overwhelm the LLM
5. **Handle Timeouts**: Set reasonable timeouts for external calls

Following these patterns will help you build robust, reliable tools that work seamlessly with your AI agents.
