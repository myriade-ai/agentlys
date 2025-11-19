import os
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import necessary google.generativeai types for mocking
from google.generativeai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    GenerationConfig,
    HarmCategory,
    Part,
    PromptFeedback,
    SafetySetting,
)
from google.generativeai.types import (
    FunctionCallingConfig as GeminiFunctionCallingConfig,
)
from google.generativeai.types import (
    FunctionDeclaration as GeminiFunctionDeclaration,
)
from google.generativeai.types import (
    Schema as GeminiSchema,
)
from google.generativeai.types import (
    Tool as GeminiTool,  # Renamed to avoid conflict with our Tool type if any
)
from google.generativeai.types import (
    Type as GeminiSDKType,  # Renamed to avoid conflict
)

from agentlys.base import AgentlysBase
from agentlys.model import Image as AgentlysImage
from agentlys.model import Message, MessagePart

# SUT imports
from agentlys.providers.gemini import (
    GeminiProvider,
    _agentlys_schema_to_gemini_schema,
    from_gemini_object,
    message_to_gemini_dict,
    parts_to_gemini_dict,
)


@pytest.fixture
def mock_chat_base(tmp_path):
    """Fixture for a mocked AgentlysBase."""
    chat = Mock(spec=AgentlysBase)
    chat.messages = []
    chat.instruction = None
    chat.context = None
    chat.functions_schema = None
    chat.use_tools_only = False
    chat.working_dir = tmp_path  # For any file operations if they were part of provider
    return chat


@pytest.fixture
def mock_agentlys_image():
    """Fixture for a mocked AgentlysImage."""
    image = Mock(spec=AgentlysImage)
    image.format = "png"
    image.to_base64 = Mock(return_value="dummy_base64_string")
    return image


# --- Test Initialization ---
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
def test_gemini_provider_init_with_api_key(
    mock_configure, mock_generative_model, mock_chat_base
):
    """Test GeminiProvider initialization with an API key argument."""
    api_key = "test_api_key"
    provider = GeminiProvider(chat=mock_chat_base, model="gemini-pro", api_key=api_key)
    mock_configure.assert_called_once_with(api_key=api_key)
    mock_generative_model.assert_called_once_with(
        model_name="gemini-pro", system_instruction=None
    )
    assert provider.chat == mock_chat_base
    assert provider.model == "gemini-pro"


@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
@patch.dict(os.environ, {"GEMINI_API_KEY": "env_api_key"})
def test_gemini_provider_init_with_env_var(
    mock_configure, mock_generative_model, mock_chat_base
):
    """Test GeminiProvider initialization with GEMINI_API_KEY environment variable."""
    provider = GeminiProvider(chat=mock_chat_base, model="gemini-pro")
    mock_configure.assert_called_once_with(api_key="env_api_key")
    mock_generative_model.assert_called_once()  # system_instruction will be None by default
    assert provider.chat == mock_chat_base


@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
@patch.dict(os.environ, {}, clear=True)  # Ensure GEMINI_API_KEY is not set
def test_gemini_provider_init_no_api_key_raises_value_error(
    mock_configure, mock_generative_model, mock_chat_base
):
    """Test GeminiProvider raises ValueError if no API key is found."""
    with pytest.raises(ValueError, match="Gemini API key must be provided"):
        GeminiProvider(chat=mock_chat_base, model="gemini-pro")


@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
def test_gemini_provider_init_with_instruction(
    mock_configure, mock_generative_model, mock_chat_base
):
    """Test GeminiProvider initialization with system instruction."""
    mock_chat_base.instruction = "Be a helpful assistant."
    _ = GeminiProvider(chat=mock_chat_base, model="gemini-pro", api_key="test_key")

    # Check that GenerativeModel was called with system_instruction (Content object)
    args, kwargs = mock_generative_model.call_args
    assert kwargs["model_name"] == "gemini-pro"
    system_instruction_arg = kwargs["system_instruction"]
    assert isinstance(system_instruction_arg, Content)
    assert len(system_instruction_arg.parts) == 1
    assert system_instruction_arg.parts[0].text == "Be a helpful assistant."
    assert system_instruction_arg.role == "user"  # As per current implementation


# --- Test Message Preparation ---


class TestMessagePreparation:
    def test_parts_to_gemini_dict_text(self):
        part = MessagePart(type="text", content="Hello")
        assert parts_to_gemini_dict(part) == {"text": "Hello"}

    def test_parts_to_gemini_dict_image(self, mock_agentlys_image):
        part = MessagePart(type="image", image=mock_agentlys_image)
        mock_agentlys_image.format = "jpeg"  # Test different format
        mock_agentlys_image.to_base64.return_value = "base64_jpeg_string"
        assert parts_to_gemini_dict(part) == {
            "inline_data": {"mime_type": "image/jpeg", "data": "base64_jpeg_string"}
        }

    def test_parts_to_gemini_dict_function_call(self):
        fc_data = {"name": "get_weather", "arguments": {"location": "Boston"}}
        part = MessagePart(type="function_call", function_call=fc_data)
        assert parts_to_gemini_dict(part) == {
            "function_call": {"name": "get_weather", "args": {"location": "Boston"}}
        }

    def test_parts_to_gemini_dict_function_result(self):
        part = MessagePart(
            type="function_result", name="get_weather", content={"weather": "sunny"}
        )
        assert parts_to_gemini_dict(part) == {
            "function_response": {
                "name": "get_weather",
                "response": {"name": "get_weather", "content": {"weather": "sunny"}},
            }
        }

    def test_message_to_gemini_dict_user_text(self):
        msg = Message(role="user", parts=[MessagePart(type="text", content="Hi there")])
        assert message_to_gemini_dict(msg) == {
            "role": "user",
            "parts": [{"text": "Hi there"}],
        }

    def test_message_to_gemini_dict_assistant_with_function_call(self):
        fc_data = {"name": "run_code", "arguments": {"code": "print('hello')"}}
        msg = Message(
            role="assistant",
            parts=[MessagePart(type="text", content="Sure, I can run that.")],
            function_call=fc_data,
        )
        expected_parts = [
            {"text": "Sure, I can run that."},
            {"function_call": {"name": "run_code", "args": {"code": "print('hello')"}}},
        ]
        assert message_to_gemini_dict(msg) == {"role": "model", "parts": expected_parts}

    def test_message_to_gemini_dict_function_response(self):
        fr_part = MessagePart(
            type="function_result", name="run_code", content={"status": "success"}
        )
        msg = Message(role="function", parts=[fr_part])
        expected_gemini_part = {
            "function_response": {
                "name": "run_code",
                "response": {"name": "run_code", "content": {"status": "success"}},
            }
        }
        assert message_to_gemini_dict(msg) == {
            "role": "function",
            "parts": [expected_gemini_part],
        }

    def test_message_to_gemini_dict_system_message_conversion(self):
        msg = Message(
            role="system",
            parts=[MessagePart(type="text", content="You are an expert.")],
        )
        # Current implementation converts system to user with prefix
        assert message_to_gemini_dict(msg) == {
            "role": "user",
            "parts": [{"text": "[System Instruction] You are an expert."}],
        }


# --- Test Schema Conversion ---
class TestSchemaConversion:
    def test_agentlys_schema_to_gemini_simple(self):
        ac_schema = {"type": "string", "description": "A simple string."}
        gemini_schema = _agentlys_schema_to_gemini_schema(ac_schema)
        assert gemini_schema.type == GeminiSDKType.STRING
        assert gemini_schema.description == "A simple string."

    def test_agentlys_schema_to_gemini_object(self):
        ac_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }
        gemini_schema = _agentlys_schema_to_gemini_schema(ac_schema)
        assert gemini_schema.type == GeminiSDKType.OBJECT
        assert gemini_schema.required == ["location"]
        assert "location" in gemini_schema.properties
        assert gemini_schema.properties["location"].type == GeminiSDKType.STRING
        assert gemini_schema.properties["location"].description == "City name"
        assert "unit" in gemini_schema.properties
        assert gemini_schema.properties["unit"].enum == ["celsius", "fahrenheit"]

    def test_agentlys_schema_to_gemini_array(self):
        ac_schema = {
            "type": "array",
            "items": {"type": "number", "description": "A number"},
        }
        gemini_schema = _agentlys_schema_to_gemini_schema(ac_schema)
        assert gemini_schema.type == GeminiSDKType.ARRAY
        assert gemini_schema.items.type == GeminiSDKType.NUMBER
        assert gemini_schema.items.description == "A number"

    def test_agentlys_schema_default_object_type(self):
        # Test when "type": "object" is missing but properties are present
        ac_schema = {"properties": {"name": {"type": "string"}}}
        gemini_schema = _agentlys_schema_to_gemini_schema(ac_schema)
        assert gemini_schema.type == GeminiSDKType.OBJECT  # Should default to OBJECT
        assert "name" in gemini_schema.properties
        assert gemini_schema.properties["name"].type == GeminiSDKType.STRING


# --- Test Response Parsing ---
class TestResponseParsing:
    def test_from_gemini_object_text_response(self):
        mock_candidate_part = Mock(spec=Part)
        mock_candidate_part.text = "Hello from Gemini"
        mock_candidate_part.function_call = None  # Ensure function_call is None

        mock_candidate = Mock(spec=Candidate)
        mock_candidate.content = Mock(spec=Content, parts=[mock_candidate_part])

        message_id = str(uuid.uuid4())
        msg = from_gemini_object(mock_candidate, id=message_id)

        assert msg.role == "assistant"
        assert msg.content == "Hello from Gemini"
        assert msg.function_call is None
        assert msg.id == message_id
        assert msg.function_call_id is None

    def test_from_gemini_object_function_call_response(self):
        mock_fc = Mock(spec=FunctionCall)
        mock_fc.name = "get_weather"
        mock_fc.args = {"location": "Tokyo"}

        mock_candidate_part_fc = Mock(spec=Part)
        mock_candidate_part_fc.text = None
        mock_candidate_part_fc.function_call = mock_fc

        # Gemini might also include a text part, let's simulate that
        mock_candidate_part_text = Mock(spec=Part)
        mock_candidate_part_text.text = "Okay, I will get the weather."
        mock_candidate_part_text.function_call = None

        mock_candidate = Mock(spec=Candidate)
        # Order of parts might matter or vary, test with FC first
        mock_candidate.content = Mock(
            spec=Content, parts=[mock_candidate_part_text, mock_candidate_part_fc]
        )

        message_id = str(uuid.uuid4())
        msg = from_gemini_object(mock_candidate, id=message_id)

        assert msg.role == "assistant"
        assert (
            msg.content == "Okay, I will get the weather."
        )  # Text content should be aggregated
        assert msg.function_call == {
            "name": "get_weather",
            "arguments": {"location": "Tokyo"},
        }
        assert msg.id == message_id
        assert msg.function_call_id == message_id


# --- Test fetch_async ---
@pytest.mark.asyncio
@patch(
    "google.generativeai.GenerativeModel"
)  # Mock at the class level for provider instance
@patch("google.generativeai.configure")  # Still needed for init
async def test_fetch_async_basic_chat(
    mock_configure, MockGenerativeModel, mock_chat_base
):
    # Setup mock client and its response
    mock_client_instance = MockGenerativeModel.return_value

    mock_response_part = Mock(spec=Part)
    mock_response_part.text = "This is a test response."
    mock_response_part.function_call = None
    mock_response_candidate = Mock(spec=Candidate)
    mock_response_candidate.content = Mock(spec=Content, parts=[mock_response_part])
    mock_gemini_response = Mock(
        spec=GenerateContentResponse,
        candidates=[mock_response_candidate],
        prompt_feedback=None,
    )
    mock_client_instance.generate_content_async = AsyncMock(
        return_value=mock_gemini_response
    )

    # Initialize provider
    provider = GeminiProvider(
        chat=mock_chat_base, model="gemini-pro", api_key="fake_key"
    )

    # Prepare chat messages
    mock_chat_base.messages = [Message(role="user", content="Hello")]

    # Call fetch_async
    result_message = await provider.fetch_async(temperature=0.7)

    # Assertions
    mock_client_instance.generate_content_async.assert_called_once()
    call_args = mock_client_instance.generate_content_async.call_args
    assert call_args.kwargs["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ]
    assert call_args.kwargs["tools"] is None
    assert call_args.kwargs["tool_config"] is None
    assert isinstance(call_args.kwargs["generation_config"], GenerationConfig)
    assert call_args.kwargs["generation_config"].temperature == 0.7

    assert result_message.content == "This is a test response."
    assert result_message.role == "assistant"
    assert result_message.id is not None


@pytest.mark.asyncio
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
async def test_fetch_async_with_function_calling(
    mock_configure, MockGenerativeModel, mock_chat_base
):
    mock_client_instance = MockGenerativeModel.return_value

    # Mock Gemini response that includes a function call
    mock_fc_response = Mock(
        spec=FunctionCall, name="get_city_weather", args={"city": "London"}
    )
    mock_response_part = Mock(spec=Part, text=None, function_call=mock_fc_response)
    mock_response_candidate = Mock(
        spec=Candidate, content=Mock(spec=Content, parts=[mock_response_part])
    )
    mock_gemini_response = Mock(
        spec=GenerateContentResponse,
        candidates=[mock_response_candidate],
        prompt_feedback=None,
    )
    mock_client_instance.generate_content_async = AsyncMock(
        return_value=mock_gemini_response
    )

    # Setup chat with function schema
    mock_chat_base.functions_schema = [
        {
            "name": "get_city_weather",
            "description": "Gets the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name."}
                },
                "required": ["city"],
            },
        }
    ]
    mock_chat_base.messages = [
        Message(role="user", content="What's the weather in London?")
    ]

    provider = GeminiProvider(
        chat=mock_chat_base, model="gemini-pro", api_key="fake_key"
    )
    result_message = await provider.fetch_async()

    # Assertions
    mock_client_instance.generate_content_async.assert_called_once()
    call_args = mock_client_instance.generate_content_async.call_args

    # Verify tools
    assert call_args.kwargs["tools"] is not None
    assert len(call_args.kwargs["tools"]) == 1
    tool_arg = call_args.kwargs["tools"][0]
    assert isinstance(tool_arg, GeminiTool)
    assert len(tool_arg.function_declarations) == 1
    fd_arg = tool_arg.function_declarations[0]
    assert isinstance(fd_arg, GeminiFunctionDeclaration)
    assert fd_arg.name == "get_city_weather"
    assert isinstance(fd_arg.parameters, GeminiSchema)  # Check if conversion happened
    assert fd_arg.parameters.type == GeminiSDKType.OBJECT

    # Verify tool_config (default AUTO mode)
    assert call_args.kwargs["tool_config"] is not None
    tool_config_arg = call_args.kwargs["tool_config"]
    assert isinstance(tool_config_arg, GeminiTool)  # Gemini uses Tool for this
    assert (
        tool_config_arg.function_calling_config.mode
        == GeminiFunctionCallingConfig.Mode.AUTO
    )

    assert result_message.role == "assistant"
    assert result_message.function_call == {
        "name": "get_city_weather",
        "arguments": {"city": "London"},
    }
    assert result_message.function_call_id == result_message.id


@pytest.mark.asyncio
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
async def test_fetch_async_use_tools_only(
    mock_configure, MockGenerativeModel, mock_chat_base
):
    mock_client_instance = MockGenerativeModel.return_value
    mock_client_instance.generate_content_async = AsyncMock(
        return_value=Mock(spec=GenerateContentResponse, candidates=[])
    )  # No specific response needed for this check

    mock_chat_base.functions_schema = [
        {"name": "dummy_func", "parameters": {"type": "object", "properties": {}}}
    ]
    mock_chat_base.use_tools_only = True  # Key setting for this test

    provider = GeminiProvider(
        chat=mock_chat_base, model="gemini-pro", api_key="fake_key"
    )
    await provider.fetch_async()

    mock_client_instance.generate_content_async.assert_called_once()
    call_args = mock_client_instance.generate_content_async.call_args
    assert (
        call_args.kwargs["tool_config"].function_calling_config.mode
        == GeminiFunctionCallingConfig.Mode.ANY
    )


@pytest.mark.asyncio
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
async def test_fetch_async_api_error(
    mock_configure, MockGenerativeModel, mock_chat_base
):
    mock_client_instance = MockGenerativeModel.return_value
    mock_client_instance.generate_content_async = AsyncMock(
        side_effect=Exception("Network Error")
    )

    provider = GeminiProvider(
        chat=mock_chat_base, model="gemini-pro", api_key="fake_key"
    )
    mock_chat_base.messages = [Message(role="user", content="Test")]

    result_message = await provider.fetch_async()

    assert result_message.role == "assistant"
    assert "Gemini API Error: Exception - Network Error" in result_message.content
    assert result_message.id is not None


@pytest.mark.asyncio
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
async def test_fetch_async_no_candidates(
    mock_configure, MockGenerativeModel, mock_chat_base
):
    mock_client_instance = MockGenerativeModel.return_value

    # Mock response with no candidates and prompt feedback
    mock_prompt_feedback = Mock(spec=PromptFeedback)
    mock_prompt_feedback.block_reason = PromptFeedback.BlockReason.SAFETY
    mock_prompt_feedback.block_reason_message = "Content blocked due to safety."
    mock_prompt_feedback.safety_ratings = [
        Mock(
            category=HarmCategory.HARM_CATEGORY_SEXUAL,
            probability=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        )
    ]  # Using actual enum members

    mock_gemini_response = Mock(
        spec=GenerateContentResponse,
        candidates=[],
        prompt_feedback=mock_prompt_feedback,
    )
    mock_client_instance.generate_content_async = AsyncMock(
        return_value=mock_gemini_response
    )

    provider = GeminiProvider(
        chat=mock_chat_base, model="gemini-pro", api_key="fake_key"
    )
    mock_chat_base.messages = [Message(role="user", content="Risky prompt")]

    result_message = await provider.fetch_async()

    assert result_message.role == "assistant"
    assert "No response candidates from model." in result_message.content
    assert "Reason: SAFETY" in result_message.content  # Check for enum name
    assert "Content blocked due to safety." in result_message.content
    assert (
        "HARM_CATEGORY_SEXUAL: BLOCK_LOW_AND_ABOVE" in result_message.content
    )  # Check for enum names
    assert result_message.id is not None


# Helper for AsyncMock if not available (e.g. older Python/unittest.mock)
if not hasattr(MagicMock, "assert_awaited_once"):

    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)

# This is a basic structure. More detailed assertions and edge cases can be added.
# For example, testing different combinations of parameters for GenerationConfig,
# more complex schema conversions, and different orderings of parts in Gemini responses.
# Also, the mocking of google.generativeai.types might need to be more precise
# if the SUT relies on specific attributes or methods of those type instances beyond basic data holding.
# For instance, if `GeminiSchema` objects created by `_agentlys_schema_to_gemini_schema`
# were then used in some logic that calls their methods, those methods would also need mocking on the mock `GeminiSchema`.
# However, here they are primarily data containers passed to `FunctionDeclaration`.

# To make GeminiSDKType.OBJECT etc. work, they need to be proper enum members.
# Actual google.generativeai.types.Type might be an enum.
# For the purpose of this test, we can mock them if direct comparison is needed,
# or ensure that the SUT code uses them correctly (e.g. by checking the string value if that's what's used).
# The current _agentlys_schema_to_gemini_schema uses getattr(GeminiSDKType, gemini_type_str.upper(), ...),
# so the mock needs to support that, or we use real GeminiSDKType if available and simple.
# For the test, `GeminiSDKType.STRING` etc. are used directly, assuming they are valid references.
# If `GeminiSDKType` is an enum, `GeminiSDKType.STRING` is an enum member, not a string "STRING".
# The test `test_agentlys_schema_to_gemini_object` checks `gemini_schema.properties["location"].type == GeminiSDKType.STRING`
# This implies GeminiSDKType.STRING should be the actual enum member, not a string.
# The `google.generativeai.types` are imported, so this should work if they behave like enums or comparable constants.

# Correcting mock for GeminiSDKType if it's an enum
# If GeminiSDKType is an enum:
# GeminiSDKType.STRING would be something like <Type.STRING: 1>
# The SUT uses getattr(GeminiSDKType, gemini_type_str.upper(), ...)
# This means GeminiSDKType needs to have attributes STRING, OBJECT etc.
# The actual `google.generativeai.types.Type` is indeed an enum.

# For PromptFeedback.BlockReason.SAFETY and HarmCategory / HarmBlockThreshold,
# using the actual enum members from the library is best.
# The mocks are defined as:
# mock_prompt_feedback.block_reason = PromptFeedback.BlockReason.SAFETY (Correct)
# mock_prompt_feedback.safety_ratings = [Mock(category=HarmCategory.HARM_CATEGORY_SEXUAL, probability=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE)] (Correct)
# The assertion checks for their .name attribute, which is standard for enums.
# `hasattr(..., 'name')` is a good check for robustly getting the name.

# AsyncMock for older Python versions
# For Python 3.8+, unittest.mock.AsyncMock is standard.
# If tests are run in older environments, a custom AsyncMock or a library like `asyncmock` might be needed.
# The provided snippet for AsyncMock is a simple version.
# For this solution, assuming Python 3.8+ where AsyncMock is available.
# If `unittest.mock.AsyncMock` is not found, the custom one will be used.
# The provided `AsyncMock` is a simple version for older Python versions.
# If using Python 3.8+, `from unittest.mock import AsyncMock` is preferred.
# Let's assume the testing environment has `unittest.mock.AsyncMock`.
