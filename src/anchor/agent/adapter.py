"""The reference agent behind the RAGAgent interface.

This is the only file where the two halves of the repo touch: the LangGraph
agent on one side, the harness's AgentRun contract on the other.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from anchor.agent.graph import build_agent
from anchor.agent.retriever import LexicalRetriever
from anchor.agent.state import initial_state
from anchor.claim import Chunk
from anchor.interface import AgentRun


class FinancialResearchAgent:
    """Financial-research RAG agent over an in-memory corpus.

    Satisfies the RAGAgent protocol. An empty answer or empty retrieval
    fails AgentRun validation at this boundary rather than confusing the
    harness downstream.
    """

    def __init__(
        self,
        corpus: Sequence[Chunk],
        model: Callable[[str], str],
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        k: int = 4,
    ) -> None:
        self._graph = build_agent(LexicalRetriever(corpus, k=k), model, checkpointer)

    def run(self, question: str, thread_id: str = "default") -> AgentRun:
        config = {"configurable": {"thread_id": thread_id}}
        final = self._graph.invoke(initial_state(question), config)
        return AgentRun(answer=final["answer"], retrieved=final["retrieved"])
