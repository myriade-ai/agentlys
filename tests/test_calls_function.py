import pytest
from agentlys import Agentlys, APIProvider

BOOKS = {
    "The Great Gatsby": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "To Kill a Mockingbird": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "1984": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
}


def list_books():
    return list(BOOKS.keys())


def read_book(book_name: str):
    return BOOKS[book_name]


TEST_CASES = [
    (
        APIProvider.ANTHROPIC,
        "claude-3-7-sonnet-latest",
    ),
    (APIProvider.OPENAI, "gpt-4o"),
    (APIProvider.OPENAI, "o1-2024-12-17"),
]


@pytest.mark.parametrize("provider, model", TEST_CASES)
@pytest.mark.vcr
def test_function_calling(provider: APIProvider, model: str):
    agent = Agentlys(
        instruction="You are a helpful assistant that can read books.",
        provider=provider,
        model=model,
    )
    agent.add_function(list_books)
    agent.add_function(read_book)

    user_query = "Hello, read book 'the great gatsby'"
    messages = []
    for message in agent.run_conversation(user_query):
        print(message.to_markdown())  # For debugging
        messages.append(message)
