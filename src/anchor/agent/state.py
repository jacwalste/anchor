"""LangGraph state schema for the reference agent.

Accumulating fields carry explicit reducers; current-state fields overwrite.
A missing reducer on an accumulating field means silent overwrites — the
classic LangGraph failure — so every list/counter here is annotated.
"""

import operator
from typing import Annotated, TypedDict

from anchor.claim import Chunk


def merge_chunks(existing: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    """Accumulate retrieved chunks across rounds, deduplicated by chunk id."""
    seen = {chunk.id for chunk in existing}
    return existing + [chunk for chunk in new if chunk.id not in seen]


class AgentState(TypedDict):
    question: str
    queries: Annotated[list[str], operator.add]  # every retrieval query tried
    retrieved: Annotated[list[Chunk], merge_chunks]  # accumulated evidence
    rounds: Annotated[int, operator.add]  # retrieval rounds completed
    sufficient: bool
    answer: str


def initial_state(question: str) -> AgentState:
    """Seed state for one run: the question is also the first retrieval query."""
    if not question.strip():
        raise ValueError("cannot run the agent on an empty question")
    return AgentState(
        question=question,
        queries=[question],
        retrieved=[],
        rounds=0,
        sufficient=False,
        answer="",
    )
