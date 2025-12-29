import unittest

from agentlys.chat import Agentlys


class MockTool:
    def __init__(self, name):
        self.name = name
        self.call_count = 0

    def __repr__(self):
        return f"MockTool(name={self.name}, call_count={self.call_count})"

    def increment(self):
        self.call_count += 1
        return self.call_count


class TestToolRepr(unittest.TestCase):
    def test_tool_repr_in_last_tools_states(self):
        agent = Agentlys(provider="openai")
        mock_tool = MockTool("TestTool")
        class_name = mock_tool.__class__.__name__
        tool_id = agent.add_tool(mock_tool)

        last_tools_states = agent.last_tools_states

        assert (
            f"### {class_name}-{tool_id}\nMockTool(name=TestTool, call_count=0)"
            in last_tools_states
        )

        # Increment the mock tool's call count
        agent.functions[f"MockTool-{tool_id}__increment"]()

        last_tools_states = agent.last_tools_states
        assert (
            f"### {class_name}-{tool_id}\nMockTool(name=TestTool, call_count=1)"
            in last_tools_states
        )

    def test_last_tools_states_in_openai_system_last_message(self):
        """
        We want to check that the last_tools_states is in the system last_message
        """
        agent = Agentlys(provider="openai")
        mock_tool = MockTool("TestTool")
        class_name = mock_tool.__class__.__name__
        tool_id = agent.add_tool(mock_tool)

        # Mock the client.messages.create method
        def mock_create(*args, **kwargs):
            # Check if the tool representation is in the system message
            messages = kwargs.get("messages", [])
            system_messages = [msg for msg in messages if msg.get("role") == "system"]

            # Verify the last_tools_states is in the system message content
            tool_repr = (
                f"### {class_name}-{tool_id}\nMockTool(name=TestTool, call_count=0)"
            )

            found = False
            for msg in system_messages:
                if tool_repr in msg["content"][0]["text"]:
                    found = True
                    break

            self.assertTrue(
                found, f"Tool representation '{tool_repr}' not found in system messages"
            )

            class MockResponseMessage:
                role = "assistant"
                content = "mock"
                tool_calls = None

            class MockResponseChoice:
                message = MockResponseMessage

            # Return a mock response
            class MockResponse:
                id = None
                choices = [MockResponseChoice]

            return MockResponse()

        # Replace the create method with our mock
        agent.provider.client.chat.completions.create = mock_create
        agent.ask("Hello")

    def test_last_tools_states_in_anthropic_system_last_message(self):
        """
        We want to check that the last_tools_states is in the system messages
        """
        agent = Agentlys(provider="anthropic")
        mock_tool = MockTool("TestTool")
        class_name = mock_tool.__class__.__name__
        tool_id = agent.add_tool(mock_tool)

        # Mock the client.messages.create method
        async def mock_create(*args, **kwargs):
            # Check if the tool representation is in the system message
            system_messages = kwargs.get("system", [])
            # Verify the last_tools_states is in the system message last content
            tool_repr = (
                f"### {class_name}-{tool_id}\nMockTool(name=TestTool, call_count=0)"
            )
            found = False
            for msg in system_messages:
                if tool_repr in msg["text"]:
                    found = True
                    break

            self.assertTrue(
                found, f"Tool representation '{tool_repr}' not found in system messages"
            )

            # Return a mock response
            class MockResponse:
                def to_dict(self):
                    return {"role": "user", "content": "mock"}

            return MockResponse()

        # Replace the create method with our mock
        agent.provider.client.messages.create = mock_create
        agent.ask("Hello")


if __name__ == "__main__":
    unittest.main()
