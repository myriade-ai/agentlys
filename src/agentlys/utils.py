import ast
import asyncio
import csv
import inspect
import re
import typing
import warnings
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
                    raise ValueError(
                        "Too many fields to display data within the character limit."
                    )
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
        try:
            limited_data = limit_data_size(data, character_limit=character_limit)
        except ValueError as e:
            return f"Error: {e}"
    else:
        limited_data = data

    # Use the union of keys across rows: heterogeneous dicts are a routine
    # shape for query results, and DictWriter raises on keys missing from
    # fieldnames.
    header = []
    seen_keys = set()
    for row in limited_data:
        for key in row:
            if key not in seen_keys:
                seen_keys.add(key)
                header.append(key)
    with StringIO() as output:
        writer = csv.DictWriter(output, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(limited_data)
        output = output.getvalue().strip()
        output = output.replace("\r\n", "\n").replace("\r", "\n")

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
        value = keyword.value
        if isinstance(value, ast.Constant):
            arguments[keyword.arg] = value.value
        elif isinstance(value, ast.List):
            arguments[keyword.arg] = [
                el.value for el in value.elts if isinstance(el, ast.Constant)
            ]
        elif isinstance(value, ast.Dict):
            arguments[keyword.arg] = {
                k.value: v.value
                for k, v in zip(value.keys, value.values)
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
            }

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
    # Prepend \n so the first "## " at the start of the file is also matched.
    # Using "\n## " (not "## ") avoids splitting on "### " subsection headings
    # which contain "## " as a substring.
    pairs = ("\n" + string).split("\n## ")[1:]

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
AllowNonTypedParamsConfig = ConfigDict(
    arbitrary_types_allowed=True,
    # Function arguments are ultimately expanded as ``func(**arguments)``.
    # Close the root object so providers know that invented keyword arguments
    # are invalid instead of letting them reach Python as a TypeError.
    extra="forbid",
)


class OmitClassJsonSchema(GenerateJsonSchema):
    def handle_invalid_for_json_schema(self, schema, error_info: str):
        raise PydanticOmit


# Google-style Args entry: `name: description` or `name (type): description`
_ARGS_PARAM_RE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")

# Section headers that end the Args block when they appear where a new
# parameter entry would be expected.
_DOCSTRING_SECTIONS = frozenset(
    [
        "returns",
        "return",
        "raises",
        "yields",
        "yield",
        "examples",
        "example",
        "note",
        "notes",
        "attributes",
    ]
)


def _parse_docstring(func_name, docstring, signature_params):
    """Extract (description, param_descriptions) from a Google-style docstring.

    The Args block is parsed indentation-aware: a line indented deeper than
    the parameter entry it follows is accumulated into that parameter's
    description. `(type)` suffixes are stripped from parameter names, and
    names are validated against the function signature — a documented
    parameter missing from the signature (or an unattributable line) emits
    a warning instead of silently corrupting the schema.
    """
    description_lines = []
    param_descriptions = {}
    in_args = False
    current_param = None  # parameter whose description is being accumulated
    current_indent = 0

    for raw_line in inspect.cleandoc(docstring).split("\n"):
        line = raw_line.strip()
        if not in_args:
            if line.lower().startswith("args:"):
                in_args = True
            elif line:  # Only add non-empty lines to description
                description_lines.append(line)
            continue

        if not line:
            break
        indent = len(raw_line) - len(raw_line.lstrip())
        if current_param is not None and indent > current_indent:
            # Indented continuation line of the current parameter entry
            if current_param in param_descriptions:
                param_descriptions[current_param] = (
                    param_descriptions[current_param] + " " + line
                ).strip()
            continue

        match = _ARGS_PARAM_RE.match(line)
        if match:
            param_name, param_desc = match.groups()
            lookup_name = param_name.lstrip("*")
            if lookup_name in signature_params:
                current_param = lookup_name
                current_indent = indent
                param_descriptions[lookup_name] = param_desc.strip()
                continue
            if param_name.lower() in _DOCSTRING_SECTIONS:
                break  # End of the Args block (Returns:, Raises:, ...)
            warnings.warn(
                f"Docstring of `{func_name}` documents parameter "
                f"'{param_name}' which is not in the function signature; "
                "it is ignored.",
                stacklevel=3,
            )
            # Consume its continuation lines without attributing them
            current_param = param_name
            current_indent = indent
        else:
            warnings.warn(
                f"Docstring of `{func_name}`: cannot attribute line "
                f"{line!r} in the Args block to a parameter; it is ignored.",
                stacklevel=3,
            )

    description = " ".join(description_lines) if description_lines else None
    return description, param_descriptions


def inspect_schema(f):
    kw = {}
    param_descriptions = {}
    signature = inspect.signature(f)

    # Parse docstring and clean up the description
    description = None
    if f.__doc__:
        description, param_descriptions = _parse_docstring(
            f.__name__, f.__doc__, set(signature.parameters)
        )

    # Get function parameters
    for n, o in signature.parameters.items():
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
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # No current event loop in this thread (get_event_loop raises on
        # non-main threads, and on every thread from Python 3.14 on)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
