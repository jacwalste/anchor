"""The RAG-agent-under-test contract: what the harness needs from any agent.

The harness never imports an agent implementation — anything that can answer
a question and report what it retrieved can be evaluated.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from anchor.claim import Chunk


class AgentRun(BaseModel):
    """One run of an agent under test: the answer plus the context it retrieved.

    Retrieved context must be non-empty: an agent that answered without
    retrieving anything cannot be groundedness-scored, and that surfaces
    here, loudly, not as a mysterious downstream error.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    retrieved: list[Chunk] = Field(min_length=1)


@runtime_checkable
class RAGAgent(Protocol):
    """Anything that can answer a question and report what it retrieved."""

    def run(self, question: str) -> AgentRun: ...
