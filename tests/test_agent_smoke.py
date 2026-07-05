"""Smoke tests for the reference agent: the loop, reducers, and checkpointing.

The agent's model is a ScriptedJudge — no live calls. These prove the graph
mechanics, not answer quality.
"""

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from anchor.agent import LexicalRetriever, build_agent, initial_state
from anchor.claim import Chunk
from tests.stubs import ScriptedJudge

CORPUS = [
    Chunk(id="c1", text="Total revenue for fiscal 2025 was $4.2 billion, up 12% over fiscal 2024."),
    Chunk(id="c2", text="Operating margin declined to 18% due to increased infrastructure spend."),
    Chunk(id="c3", text="The company repurchased $500 million of common stock during the year."),
]


def sufficient() -> str:
    return json.dumps({"sufficient": True, "refined_query": None})


def insufficient(refined_query: str) -> str:
    return json.dumps({"sufficient": False, "refined_query": refined_query})


def make_agent(
    responses: list[str], checkpointer: Any = None
) -> tuple[Any, ScriptedJudge]:
    model = ScriptedJudge(responses=responses)
    graph = build_agent(LexicalRetriever(CORPUS), model.complete, checkpointer)
    return graph, model


class TestLexicalRetriever:
    def test_most_relevant_chunk_ranks_first(self) -> None:
        results = LexicalRetriever(CORPUS).retrieve("total revenue fiscal 2025")
        assert results[0].id == "c1"

    def test_no_token_overlap_returns_nothing(self) -> None:
        assert LexicalRetriever(CORPUS).retrieve("zebra photosynthesis") == []

    def test_k_bounds_result_count(self) -> None:
        results = LexicalRetriever(CORPUS, k=1).retrieve("fiscal 2025 revenue margin")
        assert len(results) == 1

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            LexicalRetriever([])


class TestAgentGraph:
    def test_answers_after_one_round_when_sufficient(self) -> None:
        graph, model = make_agent([sufficient(), "Revenue was $4.2 billion [c1]."])
        result = graph.invoke(initial_state("total revenue fiscal 2025"))
        assert result["answer"] == "Revenue was $4.2 billion [c1]."
        assert result["rounds"] == 1
        assert "c1" in [c.id for c in result["retrieved"]]
        assert len(model.prompts) == 2  # one assess, one answer

    def test_re_retrieves_with_refined_query_when_insufficient(self) -> None:
        graph, model = make_agent(
            [
                insufficient("operating margin infrastructure spend"),
                sufficient(),
                "Margin declined to 18% [c2].",
            ]
        )
        result = graph.invoke(initial_state("how did profitability develop"))
        assert result["rounds"] == 2
        assert result["queries"] == [
            "how did profitability develop",
            "operating margin infrastructure spend",
        ]
        assert "c2" in [c.id for c in result["retrieved"]]

    def test_accumulated_chunks_are_deduplicated(self) -> None:
        graph, _ = make_agent(
            [insufficient("revenue growth fiscal 2024"), sufficient(), "Answer [c1]."]
        )
        result = graph.invoke(initial_state("total revenue fiscal 2025"))
        ids = [c.id for c in result["retrieved"]]
        assert len(ids) == len(set(ids))
        assert result["rounds"] == 2

    def test_stops_after_max_rounds_and_answers_anyway(self) -> None:
        graph, model = make_agent(
            [
                insufficient("share repurchases"),
                insufficient("stock buyback program"),
                "Best-effort answer.",
            ]
        )
        result = graph.invoke(initial_state("total revenue fiscal 2025"))
        assert result["rounds"] == 2
        assert result["answer"] == "Best-effort answer."
        assert len(model.prompts) == 3  # assess, assess, answer

    def test_checkpointer_persists_final_state(self) -> None:
        graph, _ = make_agent(
            [sufficient(), "Revenue was $4.2 billion [c1]."], checkpointer=MemorySaver()
        )
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke(initial_state("total revenue fiscal 2025"), config)
        saved = graph.get_state(config)
        assert saved.values["answer"] == "Revenue was $4.2 billion [c1]."

    def test_garbage_assessment_fails_loudly(self) -> None:
        graph, _ = make_agent(["sure, that looks fine to me!"])
        with pytest.raises(ValueError, match="invalid sufficiency assessment"):
            graph.invoke(initial_state("total revenue fiscal 2025"))

    def test_rejects_empty_question(self) -> None:
        with pytest.raises(ValueError, match="empty question"):
            initial_state("   ")
