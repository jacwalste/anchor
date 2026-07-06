"""Agentic-RAG loop for the reference agent: retrieve -> assess -> answer.

Not one-shot retrieve-then-summarize: after each retrieval the agent's model
judges whether the evidence suffices and may retrieve again with a refined
query, bounded by MAX_RETRIEVAL_ROUNDS. The model is injected as a plain
callable and the checkpointer is injected too — nothing is hardcoded.
"""

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from anchor.agent.retriever import Retriever
from anchor.agent.state import AgentState
from anchor.claim import Chunk
from anchor.parsing import strip_code_fence

MAX_RETRIEVAL_ROUNDS = 2

_ASSESS_PROMPT = """\
Decide whether the retrieved context below is sufficient to answer the question.

Respond with ONLY a JSON object:
{{"sufficient": true or false, \
"refined_query": "<a better search query if insufficient, else null>"}}

Question:
{question}

Retrieved context:
{context}
"""

_ANSWER_PROMPT = """\
Answer the question using ONLY the retrieved context below. After each factual \
statement, cite the supporting chunk id in square brackets, e.g. [10k-item7-001]. \
If the context does not contain the answer, say so plainly.

Question:
{question}

Retrieved context:
{context}
"""


class _Assessment(BaseModel):
    sufficient: bool
    refined_query: str | None = None


def build_agent(
    retriever: Retriever,
    model: Callable[[str], str],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the agent graph around an injected retriever, model, and checkpointer."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        return {"retrieved": retriever.retrieve(state["queries"][-1]), "rounds": 1}

    def assess(state: AgentState) -> dict[str, Any]:
        prompt = _ASSESS_PROMPT.format(
            question=state["question"], context=_format_context(state["retrieved"])
        )
        assessment = _parse_assessment(model(prompt))
        update: dict[str, Any] = {"sufficient": assessment.sufficient}
        if not assessment.sufficient and assessment.refined_query:
            update["queries"] = [assessment.refined_query]
        return update

    def answer(state: AgentState) -> dict[str, Any]:
        prompt = _ANSWER_PROMPT.format(
            question=state["question"], context=_format_context(state["retrieved"])
        )
        return {"answer": model(prompt)}

    def route(state: AgentState) -> str:
        if state["sufficient"] or state["rounds"] >= MAX_RETRIEVAL_ROUNDS:
            return "answer"
        return "retrieve"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("assess", assess)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges("assess", route, {"retrieve": "retrieve", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)


def _parse_assessment(raw: str) -> _Assessment:
    try:
        return _Assessment.model_validate_json(strip_code_fence(raw))
    except ValidationError as exc:
        raise ValueError(f"agent model returned an invalid sufficiency assessment:\n{raw}") from exc


def _format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(nothing retrieved)"
    return "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in chunks)
