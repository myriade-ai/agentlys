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


def test_remove_tool():
    agent = Agentlys()
    tool_id = agent.add_tool(DummyTool)
    prefix = f"DummyTool-{tool_id}__"
    assert any(name.startswith(prefix) for name in agent.functions)

    agent.remove_tool(tool_id)

    assert tool_id not in agent.tools
    assert not any(f["name"].startswith(prefix) for f in agent.functions_schema)
    assert not any(name.startswith(prefix) for name in agent.functions)


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


class OtherTool:
    def other_method(self):
        pass


def test_re_add_tool_same_id_replaces_without_duplicate_schemas():
    agent = Agentlys()
    first = DummyTool()
    second = DummyTool()
    agent.add_tool(first, tool_id="editor")
    agent.add_tool(second, tool_id="editor")

    names = [f["name"] for f in agent.functions_schema]
    assert len(names) == len(set(names))
    assert agent.tools["editor"] is second
    # Dispatch points at the replacing instance's bound method
    assert agent.functions["DummyTool-editor__method1"].__self__ is second


def test_re_add_tool_different_class_drops_old_methods():
    agent = Agentlys()
    agent.add_tool(DummyTool(), tool_id="editor")
    agent.add_tool(OtherTool(), tool_id="editor")

    names = [f["name"] for f in agent.functions_schema]
    assert "OtherTool-editor__other_method" in names
    assert not any(name.startswith("DummyTool-editor__") for name in names)
    assert not any(name.startswith("DummyTool-editor__") for name in agent.functions)


def test_add_function_same_name_replaces_in_place():
    agent = Agentlys()

    def tool_a():
        return "a"

    def tool_b():
        return "b"

    agent.add_function(tool_a, {"name": "shared", "description": "a", "parameters": {}})
    agent.add_function(
        lambda: None, {"name": "other", "description": "", "parameters": {}}
    )
    agent.add_function(tool_b, {"name": "shared", "description": "b", "parameters": {}})

    names = [f["name"] for f in agent.functions_schema]
    assert names == ["shared", "other"]
    assert agent.functions_schema[0]["description"] == "b"
    assert agent.functions["shared"] is tool_b
