import ast
import asyncio
import csv
import inspect
import typing
from inspect import Parameter
from io import StringIO
from typing import Any

from pydantic import ConfigDict, Field, create_model
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit

from agentlys.model import Message


def limit_data_size(
    data: list[dict[str, str]], character_limit: int = 10000
) -> list[dict[str, str]]:
    # Helper function to get total character count for a given row
    def get_row_char_count(row: dict[str, str]) -> int:
        return (
            sum(len(str(value)) + len(str(key)) for key, value in row.items())
            + len(row)
            - 1
        )

    # Helper function to limit the characters per field in a row
    def limit_row_chars(
        row: dict[str, str], char_limit_per_field: int
    ) -> dict[str, str]:
        return {key: str(value)[:char_limit_per_field] for key, value in row.items()}

    # Initialize the resulting data list and the character counter
    result_data = []
    total_char_count = 0

    for row in data:
        # Calculate the character count for the current row
        row_char_count = get_row_char_count(row)

        # If adding this row will exceed the limit
        if total_char_count + row_char_count > character_limit:
            # If this is the only row we are processing, then limit characters per field
            if len(result_data) == 0:
                avg_chars_per_field = (character_limit - total_char_count) // len(row)
                if avg_chars_per_field < 1:
                    return "Error: Too many fields to display data within the character limit."
                limited_row = limit_row_chars(row, avg_chars_per_field)
                result_data.append(limited_row)
            break

        # Otherwise, add this row to the result and update the total character count
        result_data.append(row)
        total_char_count += row_char_count

    return result_data


def csv_dumps(data: list[dict], character_limit: typing.Optional[int] = None) -> str:
    # Dumps to CSV, with header row
    if not data:
        return "[]"

    if character_limit:
        limited_data = limit_data_size(data, character_limit=character_limit)
    else:
        limited_data = data

    header = list(data[0].keys())
    with StringIO() as output:
        writer = csv.DictWriter(output, fieldnames=header)
        writer.writeheader()
        writer.writerows(limited_data)
        output = output.getvalue().strip()
        output.replace("\r\n", "\n").replace("\r", "\n")

    csv_content = f"```csv\n{output}\n```"

    if len(limited_data) < len(data):
        csv_content += f"\n\n... {len(limited_data)} of {len(data)} rows displayed."
    return csv_content


def parse_function(text: str) -> dict:
    # Cleaning the text
    lines = text.strip().split("\n")
    text = " ".join(
        line[2:] if line.startswith(">") else line for line in lines
    )  # remove the leading ">"

    # Replacing the multiline string delimiters
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        parts[i] = f"'''{parts[i]}'''"
    text = "".join(parts)

    # Parsing the text using ast
    parsed = ast.parse(text).body[0].value

    # Check if it's a valid function call
    if not isinstance(parsed, ast.Call):
        raise ValueError("The text does not contain a valid function call.")

    # Extracting the function name
    function_name = parsed.func.id

    # Extracting the arguments
    arguments = {}
    for keyword in parsed.keywords:
        if isinstance(keyword.value, ast.Str):
            value = keyword.value.s
            arguments[keyword.arg] = value
        elif isinstance(keyword.value, ast.List):
            arguments[keyword.arg] = [el.s for el in keyword.value.elts]
        elif isinstance(keyword.value, ast.Dict):
            dict_values = {}
            for k, v in zip(keyword.value.keys, keyword.value.values):
                if isinstance(v, ast.Str):
                    dict_values[k.s] = v.s
            arguments[keyword.arg] = dict_values

    if not arguments:
        raise ValueError("The function call does not contain any arguments.")

    return {"name": function_name, "arguments": arguments}


def split_message(message):
    """Split message into content and function_call_str
    > message = "boat > flight\n> attack()"
    > split_message(message)
    > ('boat > flight', 'attack()')
    """
    lines = message.split("\n")
    content = []
    function_call_str = []

    switch = False
    for line in lines:
        if line.startswith(">"):
            switch = True
            function_call_str.append(line[1:].strip())  # Remove the leading ">"
        else:
            if switch:
                function_call_str.append(line)
            else:
                content.append(line)

    return "\n".join(content), "\n".join(function_call_str)


def parse_chat_template(filename) -> list[Message]:
    with open(filename) as f:
        string = f.read()

    # split the string by "\n## " to get a list of speaker and message pairs
    pairs = string.split("## ")[1:]

    # split each element of the resulting list by "\n" to separate the speaker and message
    pairs = [pair.split("\n", 1) for pair in pairs]

    # create a list of tuples
    examples_pairs_str = [(pair[0], pair[1].strip()) for pair in pairs]

    parsed_examples = []
    instruction = None
    for ind, example in enumerate(examples_pairs_str):
        # If first message role is a system message, extract the example
        if ind == 0 and example[0] == "system":
            instruction = example[1]
        else:
            role = example[0].strip().lower()
            message = example[1]

            content, function_call_str = split_message(message)
            if function_call_str:
                parsed_examples.append(
                    {
                        "role": role,
                        "content": content if content else None,
                        "function_call": {**parse_function(function_call_str)},
                    }
                )
            else:
                parsed_examples.append(
                    {
                        "role": role,
                        "content": message,
                    }
                )

    examples: list[Message] = []
    for ind, example in enumerate(parsed_examples):
        # Herit name from message role
        function_call_id = None
        if "function_call" in example:
            function_call_id = "example_" + str(ind)
        if example["role"] == "function":
            function_call_id = "example_" + str(ind - 1)

        message = Message(
            **example,
            name="example_" + example["role"],
            id="example_" + str(ind),
            function_call_id=function_call_id,
        )
        examples.append(message)

    return instruction, examples


# Replace the old class with a ConfigDict
AllowNonTypedParamsConfig = ConfigDict(arbitrary_types_allowed=True)


class OmitClassJsonSchema(GenerateJsonSchema):
    def handle_invalid_for_json_schema(self, schema, error_info: str):
        raise PydanticOmit


def inspect_schema(f):
    kw = {}
    param_descriptions = {}

    # Parse docstring and clean up the description
    description = None
    if f.__doc__:
        doc_lines = f.__doc__.split("\n")
        description_lines = []
        in_args = False

        for line in doc_lines:
            line = line.strip()
            if line.lower().startswith("args:"):
                in_args = True
                continue
            if in_args:
                if not line or line.lower().startswith("returns:"):
                    break
                if ":" in line:
                    param_name, param_desc = line.split(":", 1)
                    param_descriptions[param_name.strip()] = param_desc.strip()
            else:
                if line:  # Only add non-empty lines to description
                    description_lines.append(line)

        description = " ".join(description_lines) if description_lines else None

    # Get function parameters
    for n, o in inspect.signature(f).parameters.items():
        # Skip 'self' parameter and 'from_response' parameter
        if n in ["self", "from_response"]:
            continue
        annotation = o.annotation if o.annotation != Parameter.empty else Any
        default = ... if o.default == Parameter.empty else o.default

        # Create Field with description if available
        if n in param_descriptions:
            kw[n] = (
                annotation,
                Field(default=default, description=param_descriptions[n]),
            )
        else:
            kw[n] = (annotation, default)

    s = create_model(
        f"Input for `{f.__name__}`", __config__=AllowNonTypedParamsConfig, **kw
    ).model_json_schema(mode="validation", schema_generator=OmitClassJsonSchema)
    return dict(name=f.__name__, description=description, parameters=s)


def get_event_loop_or_create():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith("There is no current event loop in thread"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        else:
            raise
