"""Reference financial-research RAG agent — the subject the harness evaluates.

Requires the `agent` extra; the core harness never imports this package.
"""

try:
    import langgraph  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "the reference agent requires the 'agent' extra: pip install anchor[agent]"
    ) from exc

from anchor.agent.adapter import FinancialResearchAgent
from anchor.agent.graph import MAX_RETRIEVAL_ROUNDS, build_agent
from anchor.agent.retriever import LexicalRetriever, Retriever
from anchor.agent.state import AgentState, initial_state

__all__ = [
    "MAX_RETRIEVAL_ROUNDS",
    "AgentState",
    "FinancialResearchAgent",
    "LexicalRetriever",
    "Retriever",
    "build_agent",
    "initial_state",
]
