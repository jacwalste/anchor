"""Benchmark runner tests: overlap matching, per-case error isolation."""

import json

from anchor import Chunk, VerdictLabel, load_benchmark, load_chunks
from anchor.benchmark import run_benchmark
from anchor.dataset import BenchmarkCase, PlantedHallucination
from tests.stubs import ScriptedJudge

CORPUS = [Chunk(id="c1", text="Total revenue for fiscal 2025 was $4.2 billion.")]

CASE = BenchmarkCase(
    id="case-x",
    question="What was revenue, and what is the market position?",
    answer="Revenue was $4.2 billion in fiscal 2025. AcmeCloud leads the cloud market.",
    chunk_ids=["c1"],
    planted=[
        PlantedHallucination(
            span="AcmeCloud leads the cloud market.", expected=VerdictLabel.UNSUPPORTED
        )
    ],
)

EXTRACTION = json.dumps(
    [
        {"text": "Revenue was $4.2B in fiscal 2025", "source_text": "Revenue was $4.2 billion"},
        {"text": "AcmeCloud leads the cloud market", "source_text": "leads the cloud market"},
    ]
)


def supported_verdict() -> str:
    return json.dumps(
        {
            "label": "supported",
            "rationale": "Stated in the chunk.",
            "evidence": {"chunk_id": "c1", "quote": "was $4.2 billion"},
        }
    )


def verdict(label: str) -> str:
    return json.dumps({"label": label, "rationale": "canned rationale", "evidence": None})


def test_planted_hallucination_is_caught_by_overlapping_flagged_claim() -> None:
    judge = ScriptedJudge(responses=[EXTRACTION, supported_verdict(), verdict("unsupported")])
    result = run_benchmark([CASE], CORPUS, judge)
    assert result.caught_count == 1
    assert result.planted_count == 1
    score = result.cases[0].score
    assert score is not None and score.score == 0.5


def test_planted_hallucination_is_missed_when_judge_says_supported() -> None:
    # Judge (wrongly) supports the planted claim — quote must exist, so it
    # cites the only chunk; the harness records a MISS, it doesn't second-guess.
    wrong_support = json.dumps(
        {
            "label": "supported",
            "rationale": "Seems fine.",
            "evidence": {"chunk_id": "c1", "quote": "fiscal 2025"},
        }
    )
    judge = ScriptedJudge(responses=[EXTRACTION, supported_verdict(), wrong_support])
    result = run_benchmark([CASE], CORPUS, judge)
    assert result.caught_count == 0


def test_non_overlapping_flagged_claim_does_not_count_as_catch() -> None:
    # The revenue claim (non-overlapping) gets flagged; the planted claim is
    # supported. The planted span itself was NOT caught.
    wrong_support = json.dumps(
        {
            "label": "supported",
            "rationale": "Seems fine.",
            "evidence": {"chunk_id": "c1", "quote": "fiscal 2025"},
        }
    )
    judge = ScriptedJudge(responses=[EXTRACTION, verdict("unsupported"), wrong_support])
    result = run_benchmark([CASE], CORPUS, judge)
    assert result.caught_count == 0


def test_judge_failure_on_one_case_does_not_kill_the_run() -> None:
    other_case = BenchmarkCase(
        id="case-y", question="q?", answer="Revenue was $4.2 billion.", chunk_ids=["c1"]
    )
    judge = ScriptedJudge(
        responses=[
            "garbage that is not json",  # case-x extraction fails
            json.dumps([{"text": "Revenue was $4.2B", "source_text": "Revenue was $4.2 billion"}]),
            supported_verdict(),
        ]
    )
    result = run_benchmark([CASE, other_case], CORPUS, judge)
    assert result.errored_case_ids == ["case-x"]
    assert result.cases[1].error is None
    score = result.cases[1].score
    assert score is not None and score.score == 1.0


def test_shipped_benchmark_runs_end_to_end_with_a_scripted_judge() -> None:
    """Integration: shipped data flows through the full runner (1 case)."""
    from pathlib import Path

    data = Path(__file__).parent.parent / "data"
    corpus = load_chunks(data / "corpus" / "acmecloud_fy2025.json")
    all_cases = load_benchmark(data / "benchmark" / "cases.json")
    cases = [c for c in all_cases if c.id.startswith("case-01")]
    assert len(cases) == 1
    judge = ScriptedJudge(
        responses=[
            json.dumps(
                [
                    {
                        "text": "AcmeCloud is the market leader in cloud infrastructure software",
                        "source_text": "AcmeCloud is the market leader",
                    }
                ]
            ),
            verdict("unsupported"),
        ]
    )
    result = run_benchmark(cases, corpus, judge)
    assert result.caught_count == 1
