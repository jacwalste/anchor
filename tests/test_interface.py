"""Interface tests: the AgentRun contract, the adapter behind it, and the front door."""

import json

import pytest
from pydantic import ValidationError

from anchor import AgentRun, Chunk, RAGAgent, VerdictLabel, evaluate_answer
from anchor.agent import FinancialResearchAgent
from tests.stubs import ScriptedJudge

CORPUS = [
    Chunk(id="c1", text="Total revenue for fiscal 2025 was $4.2 billion, up 12% over fiscal 2024."),
    Chunk(id="c2", text="Operating margin declined to 18% due to increased infrastructure spend."),
]


class TestAgentRun:
    def test_rejects_empty_answer(self) -> None:
        with pytest.raises(ValidationError):
            AgentRun(answer="", retrieved=CORPUS)

    def test_rejects_empty_retrieval(self) -> None:
        with pytest.raises(ValidationError):
            AgentRun(answer="an answer", retrieved=[])


class TestFinancialResearchAgent:
    def test_satisfies_the_protocol(self) -> None:
        agent = FinancialResearchAgent(CORPUS, model=ScriptedJudge(responses=[]).complete)
        assert isinstance(agent, RAGAgent)

    def test_runs_end_to_end(self) -> None:
        model = ScriptedJudge(
            responses=[
                json.dumps({"sufficient": True, "refined_query": None}),
                "Revenue was $4.2 billion [c1].",
            ]
        )
        agent = FinancialResearchAgent(CORPUS, model=model.complete)
        run = agent.run("total revenue fiscal 2025")
        assert run.answer == "Revenue was $4.2 billion [c1]."
        assert "c1" in [chunk.id for chunk in run.retrieved]


class TestEvaluateAnswer:
    def test_extracts_and_verifies_every_claim(self) -> None:
        answer = "Revenue grew 12% in fiscal 2025. AcmeCloud leads the market."
        judge = ScriptedJudge(
            responses=[
                json.dumps(
                    [
                        {
                            "text": "Revenue grew 12% in fiscal 2025",
                            "source_text": "Revenue grew 12%",
                        },
                        {"text": "AcmeCloud leads the market", "source_text": "leads the market"},
                    ]
                ),
                json.dumps(
                    {
                        "label": "supported",
                        "rationale": "Stated directly.",
                        "evidence": {"chunk_id": "c1", "quote": "up 12%"},
                    }
                ),
                json.dumps(
                    {
                        "label": "unsupported",
                        "rationale": "No chunk mentions this.",
                        "evidence": None,
                    }
                ),
            ]
        )
        verified = evaluate_answer(answer, CORPUS, judge)
        assert [v.verdict.label for v in verified] == [
            VerdictLabel.SUPPORTED,
            VerdictLabel.UNSUPPORTED,
        ]
        # one extraction call + one verification call per claim
        assert len(judge.prompts) == 3
