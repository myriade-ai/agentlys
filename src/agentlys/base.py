import typing

from agentlys.model import Message
from agentlys.providers.base_provider import BaseProvider


class AgentlysBase:  # TODO: rename ?
    instruction: str = (None,)
    examples: typing.Union[list[Message], None]
    messages: typing.Union[list[Message], None]
    context: str
    max_interactions: int
    model: str
    provider: BaseProvider
