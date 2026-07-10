"""Concurrency behavior of evaluate_answer: real overlap, stable order, loud failures."""

import json
import threading

import pytest

from anchor import Chunk, JudgeResponseError, VerdictLabel, evaluate_answer

CONTEXT = [Chunk(id="c1", text="Total revenue for fiscal 2025 was $4.2 billion.")]
ANSWER = "Revenue was $4.2 billion. AcmeCloud leads the cloud market."

EXTRACTION = json.dumps(
    [
        {"text": "Revenue was $4.2B in fiscal 2025", "source_text": "Revenue was $4.2 billion"},
        {
            "text": "AcmeCloud leads the cloud market",
            "source_text": "AcmeCloud leads the cloud market",
        },
    ]
)
SUPPORTED = json.dumps(
    {
        "label": "supported",
        "rationale": "Stated directly.",
        "evidence": {"chunk_id": "c1", "quote": "$4.2 billion"},
    }
)
UNSUPPORTED = json.dumps({"label": "unsupported", "rationale": "Not in context.", "evidence": None})


class KeyedJudge:
    """Thread-safe judge keyed on prompt content, so call order cannot matter."""

    def __init__(self, garbage_for: str | None = None) -> None:
        self._garbage_for = garbage_for

    def complete(self, prompt: str) -> str:
        if "Decompose the answer" in prompt:
            return EXTRACTION
        if self._garbage_for is not None and self._garbage_for in prompt:
            return "definitely not json"
        if "Revenue was $4.2B in fiscal 2025" in prompt:
            return SUPPORTED
        return UNSUPPORTED


def test_results_keep_claim_order_under_concurrency() -> None:
    verified = evaluate_answer(ANSWER, CONTEXT, KeyedJudge(), max_concurrency=4)
    assert [item.verdict.label for item in verified] == [
        VerdictLabel.SUPPORTED,
        VerdictLabel.UNSUPPORTED,
    ]
    assert verified[0].claim.text == "Revenue was $4.2B in fiscal 2025"


def test_verifications_actually_overlap() -> None:
    """Both verification calls must be in flight at once, or the barrier trips."""
    barrier = threading.Barrier(2, timeout=5)

    class BarrierJudge:
        def __init__(self) -> None:
            self._sent_extraction = False
            self._lock = threading.Lock()

        def complete(self, prompt: str) -> str:
            with self._lock:
                if not self._sent_extraction:
                    self._sent_extraction = True
                    return EXTRACTION
            barrier.wait()
            return UNSUPPORTED

    verified = evaluate_answer(ANSWER, CONTEXT, BarrierJudge(), max_concurrency=2)
    assert len(verified) == 2


def test_verification_failure_propagates() -> None:
    judge = KeyedJudge(garbage_for="AcmeCloud leads the cloud market")
    with pytest.raises(JudgeResponseError, match="not valid JSON"):
        evaluate_answer(ANSWER, CONTEXT, judge, max_concurrency=4)


def test_rejects_invalid_max_concurrency() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        evaluate_answer(ANSWER, CONTEXT, KeyedJudge(), max_concurrency=0)
