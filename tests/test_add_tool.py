from agentlys.chat import Agentlys


class DummyTool:
    def method1(self):
        pass

    def method2(self, param: str):
        pass

    def _private_method(self):
        pass

    @classmethod
    def class_method(cls):
        pass


def test_add_tool():
    agent = Agentlys()
    tool_id = agent.add_tool(DummyTool)

    # Check if the tool was added correctly
    assert tool_id in agent.tools
    assert isinstance(agent.tools[tool_id], DummyTool)

    # Check if all methods (excluding private) were added to functions_schema
    function_names = [f["name"] for f in agent.functions_schema]
    assert f"DummyTool-{tool_id}__method1" in function_names
    assert f"DummyTool-{tool_id}__method2" in function_names
    assert f"DummyTool-{tool_id}___private_method" not in function_names

    # Check if all functions (excluding private) were added correctly
    assert f"DummyTool-{tool_id}__method1" in agent.functions
    assert f"DummyTool-{tool_id}__method2" in agent.functions
    assert f"DummyTool-{tool_id}___private_method" not in agent.functions

    assert f"DummyTool-{tool_id}__class_method" in agent.functions


def test_add_tool_with_custom_id():
    agent = Agentlys()
    custom_id = "custom_tool_id"
    tool_id = agent.add_tool(DummyTool(), tool_id=custom_id)

    assert tool_id == custom_id
    assert custom_id in agent.tools
    assert isinstance(agent.tools[custom_id], DummyTool)


def test_add_tool_instance():
    agent = Agentlys()
    tool_instance = DummyTool()
    tool_id = agent.add_tool(tool_instance)

    assert tool_id in agent.tools
    assert agent.tools[tool_id] is tool_instance

    # Check if all methods (excluding private) were added to functions_schema
    function_names = [f["name"] for f in agent.functions_schema]
    assert f"DummyTool-{tool_id}__method1" in function_names
    assert f"DummyTool-{tool_id}__method2" in function_names
    assert f"DummyTool-{tool_id}___private_method" not in function_names
