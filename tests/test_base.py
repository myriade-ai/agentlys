import unittest
from unittest.mock import patch

from agentlys import Agentlys, Message
from agentlys.providers.openai import OpenAIProvider


class TestAgentlys(unittest.TestCase):
    def test_agentlys_initialization(self):
        agent = Agentlys(instruction="Test instruction", provider="openai")
        self.assertEqual(agent.instruction, "Test instruction")
        self.assertEqual(agent.provider.__class__, OpenAIProvider)
        # Instead of checking for a specific model, we'll just check if it's a string
        self.assertIsInstance(agent.model, str)
        self.assertTrue(len(agent.model) > 0)  # Ensure the model is not an empty string

    def test_agentlys_invalid_provider(self):
        with self.assertRaises(ValueError):
            Agentlys(provider="invalid_provider")

    def test_add_function(self):
        agent = Agentlys()

        def test_function(arg1: str, arg2: int) -> str:
            return f"Received {arg1} and {arg2}"

        agent.add_function(test_function)
        self.assertEqual(len(agent.functions_schema), 1)
        self.assertIn("test_function", agent.functions)

    @patch.object(OpenAIProvider, "fetch_async")
    def test_ask(self, mock_fetch_openai):
        mock_fetch_openai.return_value = Message(
            role="assistant", content="Test response"
        )
        agent = Agentlys(provider="openai")

        response = agent.ask("Test question")
        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "Test response")
        self.assertEqual(len(agent.messages), 2)

    @patch.object(OpenAIProvider, "fetch_async")
    def test_run_conversation(self, mock_fetch_openai):
        mock_fetch_openai.return_value = Message(
            role="assistant", content="Final response"
        )
        agent = Agentlys(provider="openai")

        responses = list(agent.run_conversation("Test question"))
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].role, "user")
        self.assertEqual(responses[0].content, "Test question")
        self.assertEqual(responses[1].role, "assistant")
        self.assertEqual(responses[1].content, "Final response")


class TestFormatCallbackMessage(unittest.TestCase):
    def test_list_of_numbers_is_formatted_as_text(self):
        agent = Agentlys()
        message = agent._format_callback_message(
            function_name="fn",
            function_call_id="call_1",
            content=[1, 2, 3],
            image=None,
        )
        self.assertEqual(message.parts[0].content, "1\n2\n3")
        self.assertEqual(agent.tools, {})

    def test_scalar_is_formatted_as_text(self):
        agent = Agentlys()
        message = agent._format_callback_message(
            function_name="fn",
            function_call_id="call_1",
            content=42,
            image=None,
        )
        self.assertEqual(message.parts[0].content, "42")
        self.assertEqual(agent.tools, {})


class TestMcpServersConstructor(unittest.TestCase):
    def test_constructor_rejects_mcp_servers(self):
        # add_mcp_server is async; registering from __init__ silently did
        # nothing, so passing servers here must fail loudly instead.
        with self.assertRaisesRegex(ValueError, "add_mcp_server"):
            Agentlys(mcp_servers=[object()])

    def test_constructor_accepts_none(self):
        agent = Agentlys(mcp_servers=None)
        self.assertEqual(agent.messages, [])


if __name__ == "__main__":
    unittest.main()
