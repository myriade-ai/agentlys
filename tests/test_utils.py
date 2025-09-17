import unittest

from pydantic import BaseModel

from agentlys.model import Message
from agentlys.utils import inspect_schema, limit_data_size, parse_function


class TestLimitDataSizeUpdated(unittest.TestCase):
    def setUp(self):
        self.test_data_1 = [
            {"name": "Alice", "age": "25", "city": "New York"},
            {"name": "Bob", "age": "30", "city": "San Francisco"},
            {"name": "Charlie", "age": "35", "city": "Los Angeles"},
            {"name": "Daisy", "age": "40", "city": "Houston"},
        ]

        self.test_data_2 = [
            {"name": "Alice", "age": "25", "city": "New York"},
            {"name": "Bob", "age": "30", "city": "San Francisco"},
        ]

        self.test_data_3 = [
            {"name": "Alice", "age": "25", "city": "New York"},
            {"name": "B" * 50, "age": "30", "city": "San Francisco"},
        ]

    def test_small_character_limit(self):
        result = limit_data_size(self.test_data_2, character_limit=5)
        self.assertEqual(result, [{"name": "A", "age": "2", "city": "N"}])

    def test_larger_character_limit(self):
        result = limit_data_size(self.test_data_1, character_limit=150)
        expected = [
            {"name": "Alice", "age": "25", "city": "New York"},
            {"name": "Bob", "age": "30", "city": "San Francisco"},
            {"name": "Charlie", "age": "35", "city": "Los Angeles"},
            {"name": "Daisy", "age": "40", "city": "Houston"},
        ]
        self.assertEqual(result, expected)

    def test_very_long_single_field(self):
        result = limit_data_size(self.test_data_3, character_limit=80)
        expected = [{"name": "Alice", "age": "25", "city": "New York"}]
        self.assertEqual(result, expected)


plot_widget = """
> PLOT_WIDGET(
    name="Distribution of stations per city",
    outputType="Doughnut2d",
    sql="SELECT city, COUNT(*) FROM public.station GROUP BY city",
    params={"xAxisName": "City", "yAxisName":"Number of stations", "xKey":"city", "yKey":"count"}
)
"""


class TestParseFunction(unittest.TestCase):
    def test_no_arguments(self):
        text = """
        > FUNCTION()
        """
        # Except raises ValueError
        with self.assertRaises(ValueError):
            parse_function(text)

    def test_single_argument(self):
        text = """
        > FUNCTION(name="single argument")
        """
        result = parse_function(text)
        expected = {"name": "FUNCTION", "arguments": {"name": "single argument"}}
        self.assertEqual(result, expected)

    def test_multiple_arguments(self):
        text = """
        > FUNCTION(name="argument1", another="argument2")
        """
        result = parse_function(text)
        expected = {
            "name": "FUNCTION",
            "arguments": {"name": "argument1", "another": "argument2"},
        }
        self.assertEqual(result, expected)

    def test_multiple_arguments_with_newlines(self):
        text = """
        > FUNCTION(
            name="argument1",
            another="argument2"
        )
        """
        result = parse_function(text)
        expected = {
            "name": "FUNCTION",
            "arguments": {"name": "argument1", "another": "argument2"},
        }
        self.assertEqual(result, expected)

    def test_multiline_argument(self):
        text = """
        > FUNCTION(name="argument1", query=```SELECT column
        FROM "table";
        ```)
        """
        result = parse_function(text)
        expected = {
            "name": "FUNCTION",
            "arguments": {
                "name": "argument1",
                "query": 'SELECT column         FROM "table";         ',
            },
        }
        self.assertEqual(result, expected)

    def test_multiline_with_other_arguments(self):
        text = """
        > FUNCTION(name="argument1", description="describes something", query=```SELECT column
        FROM table;
        ```)
        """
        result = parse_function(text)
        expected = {
            "name": "FUNCTION",
            "arguments": {
                "name": "argument1",
                "description": "describes something",
                "query": "SELECT column         FROM table;         ",
            },
        }
        self.assertEqual(result, expected)

    def test_parse_function(self):
        text = """
        > SQL_QUERY(name="installation_date column examples", query=```SELECT installation_date
        FROM public.station
        ORDER BY RANDOM()
        LIMIT 5;
        ```)
        """
        result = parse_function(text)
        expected = {
            "name": "SQL_QUERY",
            "arguments": {
                "name": "installation_date column examples",
                "query": "SELECT installation_date         FROM public.station         ORDER BY RANDOM()         LIMIT 5;         ",
            },
        }
        self.assertEqual(result, expected)

    def test_parse_array(self):
        text = """
        > SQL_QUERY(name=["installation_date column examples", "another name"], test="one")
        """
        result = parse_function(text)
        expected = {
            "name": "SQL_QUERY",
            "arguments": {
                "name": ["installation_date column examples", "another name"],
                "test": "one",
            },
        }
        self.assertEqual(result, expected)

    def test_plot_widget(self):
        result = parse_function(plot_widget)
        expected = {
            "name": "PLOT_WIDGET",
            "arguments": {
                "name": "Distribution of stations per city",
                "outputType": "Doughnut2d",
                "sql": "SELECT city, COUNT(*) FROM public.station GROUP BY city",
                "params": {
                    "xAxisName": "City",
                    "yAxisName": "Number of stations",
                    "xKey": "city",
                    "yKey": "count",
                },
            },
        }
        self.assertDictEqual(result, expected)


def empty_function(test):  # No type hinting, no docstring
    return "empty"


def sums(a: int, b: int = 1):
    """Adds a + b"""
    return a + b


def function_with_description_and_default_value(a: int, b: int = 1):
    """Adds a + b
    Args:
        a: The first number to add
        b: The second number to add, defaults to 1
    Returns:
        The sum of a and b
    """
    return a + b


# We should ignore from_response argument
def function_with_from_response(a: int, from_response: Message):
    """
    Args:
        a: The number to return
        from_response: The message which triggered the function call
    Returns:
        The number a
    """
    return a


class RandomClass:
    def __init__(self, content: str):
        self.content = content


def function_with_class(a: int, b: RandomClass):
    """
    Args:
        a: The number to return
        b: The message which triggered the function call
    Returns:
        The number a
    """
    return a


class RandomClassInheritedFromBaseModel(BaseModel):
    def __init__(self, content: str):
        self.content = content


def function_with_class_inherited_from_base_model(
    a: int, b: RandomClassInheritedFromBaseModel
):
    """
    Args:
        a: The number to return
        b: The message which triggered the function call
    Returns:
        The number a
    """
    return a


class TestInspectSchema(unittest.TestCase):
    maxDiff = None

    def test_inspect_schema(self):
        result = inspect_schema(sums)
        expected = {
            "name": "sums",
            "description": "Adds a + b",
            "parameters": {
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "b": {"default": 1, "title": "B", "type": "integer"},
                },
                "required": ["a"],
                "title": "Input for `sums`",
                "type": "object",
            },
        }
        self.assertEqual(result, expected)

    def test_inspect_schema_empty(self):
        result = inspect_schema(empty_function)
        expected = {
            "name": "empty_function",
            "description": None,
            "parameters": {
                "properties": {
                    "test": {"title": "Test"},
                },
                "required": ["test"],
                "title": "Input for `empty_function`",
                "type": "object",
            },
        }
        self.assertEqual(result, expected)

    def test_inspect_schema_with_default_value(self):
        result = inspect_schema(function_with_description_and_default_value)
        expected = {
            "name": "function_with_description_and_default_value",
            "description": "Adds a + b",
            "parameters": {
                "properties": {
                    "a": {
                        "title": "A",
                        "type": "integer",
                        "description": "The first number to add",
                    },
                    "b": {
                        "default": 1,
                        "title": "B",
                        "type": "integer",
                        "description": "The second number to add, defaults to 1",
                    },
                },
                "required": ["a"],
                "title": "Input for `function_with_description_and_default_value`",
                "type": "object",
            },
        }
        self.assertDictEqual(result, expected)

    def test_inspect_schema_with_from_response(self):
        result = inspect_schema(function_with_from_response)
        expected = {
            "name": "function_with_from_response",
            "description": None,
            "parameters": {
                "properties": {
                    "a": {
                        "title": "A",
                        "type": "integer",
                        "description": "The number to return",
                    },
                },
                "required": ["a"],
                "title": "Input for `function_with_from_response`",
                "type": "object",
            },
        }
        self.assertEqual(result, expected)

    def test_inspect_schema_with_class(self):
        """
        Class should be omitted from the json schema
        """
        result = inspect_schema(function_with_class)
        expected = {
            "name": "function_with_class",
            "description": None,
            "parameters": {
                "properties": {
                    "a": {
                        "description": "The number to return",
                        "title": "A",
                        "type": "integer",
                    }
                },
                "required": ["a"],
                "title": "Input for `function_with_class`",
                "type": "object",
            },
        }
        self.assertEqual(result, expected)

    def test_inspect_schema_with_class_inherited_from_base_model(self):
        result = inspect_schema(function_with_class_inherited_from_base_model)
        expected = {
            "name": "function_with_class_inherited_from_base_model",
            "description": None,
            "parameters": {
                "$defs": {
                    "RandomClassInheritedFromBaseModel": {
                        "properties": {},
                        "title": "RandomClassInheritedFromBaseModel",
                        "type": "object",
                    }
                },
                "properties": {
                    "a": {
                        "title": "A",
                        "type": "integer",
                        "description": "The number to return",
                    },
                    "b": {
                        "$ref": "#/$defs/RandomClassInheritedFromBaseModel",
                        "description": "The message which triggered the function call",
                    },
                },
                "required": ["a", "b"],
                "title": "Input for `function_with_class_inherited_from_base_model`",
                "type": "object",
            },
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
