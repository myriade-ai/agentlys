import os
import typing
import uuid  # For generating unique IDs

import google.generativeai as genai
from google.generativeai.types import (
    Candidate,
    Content,
    FunctionCallingConfig,
    FunctionDeclaration,
    GenerationConfig,
    Part,
    Schema,
    Tool,
)
from google.generativeai.types import (
    Type as GeminiType,  # Renamed to avoid conflict with typing.Type
)

from agentlys.base import AgentlysBase
from agentlys.model import Message, MessagePart
from agentlys.providers.base_provider import BaseProvider


def _agentlys_schema_to_gemini_schema(agentlys_schema: dict) -> Schema:
    """Converts an Agentlys JSON-like schema to a Gemini Schema object."""
    if not isinstance(agentlys_schema, dict):
        raise ValueError(f"Schema must be a dictionary, got {type(agentlys_schema)}")

    gemini_type_str = agentlys_schema.get("type", "OBJECT").upper()
    gemini_type = getattr(GeminiType, gemini_type_str, GeminiType.TYPE_UNSPECIFIED)

    if (
        gemini_type == GeminiType.TYPE_UNSPECIFIED and gemini_type_str == "OBJECT"
    ):  # common case for missing type
        gemini_type = GeminiType.OBJECT
    elif gemini_type == GeminiType.TYPE_UNSPECIFIED and gemini_type_str == "ARRAY":
        gemini_type = GeminiType.ARRAY
    elif gemini_type == GeminiType.TYPE_UNSPECIFIED:
        # Fallback for other unspecified types, or raise error
        # For now, let's try to map 'ANY' or other potential values if they make sense in JSON schema
        # Defaulting to STRING if type is truly unknown or not directly mappable.
        # This part might need more robust handling based on typical JSON schema variations.
        print(
            f"Warning: Unknown schema type '{agentlys_schema.get('type')}'. Defaulting to STRING."
        )
        gemini_type = GeminiType.STRING

    description = agentlys_schema.get("description")
    enum_values = agentlys_schema.get("enum")

    properties_schema = None
    if gemini_type == GeminiType.OBJECT:
        properties = agentlys_schema.get("properties")
        if properties and isinstance(properties, dict):
            properties_schema = {
                key: _agentlys_schema_to_gemini_schema(value)
                for key, value in properties.items()
            }

    items_schema = None
    if gemini_type == GeminiType.ARRAY:
        items = agentlys_schema.get("items")
        if items and isinstance(items, dict):
            items_schema = _agentlys_schema_to_gemini_schema(items)
        # If items is not a dict (e.g. not further specified), it might be an array of simple types.
        # The Gemini SDK's Schema for ARRAY expects 'items' to be another Schema.
        # If 'items' is missing or not a dict, this could lead to issues.
        # For now, we only convert if it's a dict. A more robust version might handle simple type arrays.

    required_fields = agentlys_schema.get("required")

    return Schema(
        type=gemini_type,
        description=description,
        enum=enum_values,
        properties=properties_schema
        if properties_schema
        else None,  # Ensure None if empty
        items=items_schema,
        required=required_fields if required_fields else None,  # Ensure None if empty
    )


def parts_to_gemini_dict(part: MessagePart) -> dict:
    if part.type == "text":
        return {"text": part.content}
    elif part.type == "image":
        mime_type = part.image.format.lower()
        if not mime_type.startswith("image/"):
            mime_type = f"image/{mime_type}"
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": part.image.to_base64(),
            }
        }
    elif part.type == "function_call":
        return {
            "function_call": {
                "name": part.function_call["name"],
                "args": part.function_call["arguments"],
            }
        }
    elif part.type == "function_result":
        return {
            "function_response": {
                "name": part.name,
                "response": {
                    "name": part.name,
                    "content": part.content,
                },
            }
        }
    raise ValueError(f"Unsupported part type for Gemini: {part.type}")


def message_to_gemini_dict(message: Message) -> dict:
    gemini_role = ""
    parts = []

    if message.role == "system":
        gemini_role = "user"
        system_content = " ".join(
            [p.content for p in message.parts if p.type == "text" and p.content]
        )
        parts.append({"text": f"[System Instruction] {system_content}"})
    elif message.role == "user":
        gemini_role = "user"
        for part_content in message.parts:
            parts.append(parts_to_gemini_dict(part_content))
    elif message.role == "assistant":
        gemini_role = "model"
        for part_content in message.parts:
            parts.append(parts_to_gemini_dict(part_content))
        if message.function_call:
            is_fc_already_in_parts = any(
                "function_call" in part_dict for part_dict in parts
            )
            if not is_fc_already_in_parts:
                parts.append(
                    {
                        "function_call": {
                            "name": message.function_call["name"],
                            "args": message.function_call["arguments"],
                        }
                    }
                )
    elif message.role == "function":
        gemini_role = "function"
        if len(message.parts) == 1 and message.parts[0].type == "function_result":
            parts.append(parts_to_gemini_dict(message.parts[0]))
        else:
            raise ValueError(
                "A 'function' role message must contain exactly one 'function_result' part."
            )
    else:
        raise ValueError(f"Unsupported message role for Gemini: {message.role}")

    return {"role": gemini_role, "parts": parts}


def from_gemini_object(
    gemini_response_candidate: Candidate,
    role: str = "assistant",
    id: str = None,  # ID is now mandatory, generated by fetch_async
) -> Message:
    content_text_parts = []
    function_call_dict = None

    for part_from_candidate in gemini_response_candidate.content.parts:
        if part_from_candidate.text:
            content_text_parts.append(part_from_candidate.text)
        elif (
            hasattr(part_from_candidate, "function_call")
            and part_from_candidate.function_call
        ):
            function_call_dict = {
                "name": part_from_candidate.function_call.name,
                "arguments": dict(part_from_candidate.function_call.args),
            }

    full_content = "\n".join(content_text_parts) if content_text_parts else None

    message_id = id  # Use the ID passed from fetch_async
    message_function_call_id = message_id if function_call_dict else None

    return Message(
        role=role,
        content=full_content,
        function_call=function_call_dict,
        id=message_id,
        function_call_id=message_function_call_id,
    )


class GeminiProvider(BaseProvider):
    """
    A class to provide responses from Google's Gemini API.
    """

    def __init__(
        self, chat: AgentlysBase, model: str, api_key: typing.Optional[str] = None
    ):
        super().__init__()
        self.chat = chat
        self.model = model

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key must be provided either through api_key argument or GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=resolved_api_key)

        system_instruction_content = None
        if self.chat.instruction:
            system_instruction_content = Content(
                parts=[Part(text=self.chat.instruction)], role="user"
            )

        self.client = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction_content,
        )

    async def fetch_async(self, **kwargs) -> Message:
        gemini_messages_history = self.prepare_messages(
            transform_function=message_to_gemini_dict,
        )

        # Tool Preparation
        gemini_tools_list = None
        function_declarations = []
        if self.chat.functions_schema:
            for func_schema in self.chat.functions_schema:
                func_name = func_schema.get("name")
                if not func_name:
                    print(
                        f"Warning: Function schema missing name: {func_schema}. Skipping."
                    )
                    continue

                parameters_dict = func_schema.get("parameters")
                gemini_schema_for_params = None
                if (
                    isinstance(parameters_dict, dict) and parameters_dict
                ):  # Ensure it's a non-empty dict
                    try:
                        gemini_schema_for_params = _agentlys_schema_to_gemini_schema(
                            parameters_dict
                        )
                    except ValueError as e:
                        print(
                            f"Warning: Could not convert parameters for function {func_name} due to: {e}. Skipping params."
                        )
                        gemini_schema_for_params = None  # Or handle error more strictly
                elif parameters_dict is not None:
                    print(
                        f"Warning: Parameters for function {func_name} are not a dict or are empty: {parameters_dict}. Assuming no parameters."
                    )

                function_declarations.append(
                    FunctionDeclaration(
                        name=func_name,
                        description=func_schema.get("description"),
                        parameters=gemini_schema_for_params,
                    )
                )
            if function_declarations:
                gemini_tools_list = [Tool(function_declarations=function_declarations)]

        # Tool Configuration
        tool_config_object = None
        if gemini_tools_list:
            mode = FunctionCallingConfig.Mode.AUTO
            if self.chat.use_tools_only:
                mode = FunctionCallingConfig.Mode.ANY

            fcc = FunctionCallingConfig(mode=mode)
            tool_config_object = Tool(
                function_calling_config=fcc
            )  # Gemini uses Tool for tool_config

        valid_gen_config_keys = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "max_output_tokens",
            "stop_sequences",
        }
        generation_config_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in valid_gen_config_keys and v is not None
        }
        final_generation_config = (
            GenerationConfig(**generation_config_kwargs)
            if generation_config_kwargs
            else None
        )

        try:
            response = await self.client.generate_content_async(
                contents=gemini_messages_history,
                tools=gemini_tools_list,
                tool_config=tool_config_object,
                generation_config=final_generation_config,
            )
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_content = f"Gemini API Error: {type(e).__name__} - {str(e)}"
            return Message(role="assistant", content=error_content, id=error_id)

        response_message_id = str(uuid.uuid4())  # Generate ID for the incoming message

        if not response.candidates:
            feedback_info = "No response candidates from model."
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                block_reason_name = "N/A"
                if response.prompt_feedback.block_reason:
                    block_reason_name = (
                        response.prompt_feedback.block_reason.name
                        if hasattr(response.prompt_feedback.block_reason, "name")
                        else str(response.prompt_feedback.block_reason)
                    )

                safety_ratings_str_parts = []
                if response.prompt_feedback.safety_ratings:
                    for sr in response.prompt_feedback.safety_ratings:
                        category_name = (
                            sr.category.name
                            if hasattr(sr.category, "name")
                            else str(sr.category)
                        )
                        probability_name = (
                            sr.probability.name
                            if hasattr(sr.probability, "name")
                            else str(sr.probability)
                        )
                        safety_ratings_str_parts.append(
                            f"{category_name}: {probability_name}"
                        )
                safety_ratings_str = ", ".join(safety_ratings_str_parts)

                feedback_info = (
                    f"Reason: {block_reason_name}. "
                    f"Message: {response.prompt_feedback.block_reason_message or 'N/A'}. "
                    f"Safety Ratings: [{safety_ratings_str or 'N/A'}]"
                )
            # Use the generated ID for error message too
            return Message(
                role="assistant", content=feedback_info, id=response_message_id
            )

        return from_gemini_object(
            response.candidates[0], role="assistant", id=response_message_id
        )
