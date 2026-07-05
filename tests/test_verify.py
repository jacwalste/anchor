"""Verification tests: parsing, verdict validation, and evidence provenance.

These prove plumbing and parsing — NOT the context-only rule (D3), which
lives in the verification prompt and is only demonstrable with a live judge
(see PLAN.md §3).
"""

import json

import pytest

from anchor import Chunk, Claim, JudgeResponseError, VerdictLabel, verify_claim
from tests.stubs import ScriptedJudge

CLAIM = Claim(text="Acme's FY2025 revenue was $4.2B", source_text="to $4.2B in FY2025")

CONTEXT = [
    Chunk(id="10k-item7-001", text="Total revenue for fiscal 2025 was $4.2 billion."),
    Chunk(id="10k-item7-002", text="Operating margin declined to 18% in fiscal 2025."),
]


def supported_response(quote: str = "Total revenue for fiscal 2025 was $4.2 billion.") -> str:
    return json.dumps(
        {
            "label": "supported",
            "rationale": "The chunk states the revenue figure directly.",
            "evidence": {"chunk_id": "10k-item7-001", "quote": quote},
        }
    )


def test_supported_verdict_with_located_evidence() -> None:
    judge = ScriptedJudge(responses=[supported_response()])
    verdict = verify_claim(CLAIM, CONTEXT, judge)
    assert verdict.label is VerdictLabel.SUPPORTED
    assert verdict.evidence is not None
    assert verdict.evidence.chunk_id == "10k-item7-001"


def test_unsupported_verdict_with_null_evidence() -> None:
    judge = ScriptedJudge(
        responses=[
            json.dumps(
                {
                    "label": "unsupported",
                    "rationale": "No chunk mentions the claim.",
                    "evidence": None,
                }
            )
        ]
    )
    verdict = verify_claim(CLAIM, CONTEXT, judge)
    assert verdict.label is VerdictLabel.UNSUPPORTED
    assert verdict.evidence is None


def test_prompt_contains_claim_and_context() -> None:
    judge = ScriptedJudge(responses=[supported_response()])
    verify_claim(CLAIM, CONTEXT, judge)
    prompt = judge.prompts[0]
    assert CLAIM.text in prompt
    for chunk in CONTEXT:
        assert chunk.id in prompt
        assert chunk.text in prompt


def test_rejects_invalid_json() -> None:
    judge = ScriptedJudge(responses=["The claim looks supported to me."])
    with pytest.raises(JudgeResponseError, match="not valid JSON"):
        verify_claim(CLAIM, CONTEXT, judge)


def test_rejects_non_object_json() -> None:
    judge = ScriptedJudge(responses=['["supported"]'])
    with pytest.raises(JudgeResponseError, match="expected a JSON object"):
        verify_claim(CLAIM, CONTEXT, judge)


def test_rejects_unknown_verdict_label() -> None:
    judge = ScriptedJudge(
        responses=[json.dumps({"label": "plausible", "rationale": "seems fine", "evidence": None})]
    )
    with pytest.raises(JudgeResponseError, match="schema validation"):
        verify_claim(CLAIM, CONTEXT, judge)


def test_rejects_supported_without_evidence() -> None:
    judge = ScriptedJudge(
        responses=[json.dumps({"label": "supported", "rationale": "trust me", "evidence": None})]
    )
    with pytest.raises(JudgeResponseError, match="schema validation"):
        verify_claim(CLAIM, CONTEXT, judge)


def test_rejects_evidence_citing_unknown_chunk() -> None:
    response = json.dumps(
        {
            "label": "supported",
            "rationale": "The chunk states it.",
            "evidence": {"chunk_id": "no-such-chunk", "quote": "irrelevant"},
        }
    )
    judge = ScriptedJudge(responses=[response])
    with pytest.raises(JudgeResponseError, match="unknown chunk"):
        verify_claim(CLAIM, CONTEXT, judge)


def test_rejects_fabricated_quote() -> None:
    judge = ScriptedJudge(responses=[supported_response(quote="Revenue was $9.9 billion.")])
    with pytest.raises(JudgeResponseError, match="not found verbatim"):
        verify_claim(CLAIM, CONTEXT, judge)


def test_error_carries_raw_response() -> None:
    raw = "not json"
    judge = ScriptedJudge(responses=[raw])
    with pytest.raises(JudgeResponseError) as exc_info:
        verify_claim(CLAIM, CONTEXT, judge)
    assert exc_info.value.raw_response == raw


def test_rejects_empty_context_without_calling_judge() -> None:
    judge = ScriptedJudge(responses=[])
    with pytest.raises(ValueError, match="empty context"):
        verify_claim(CLAIM, [], judge)
    assert judge.prompts == []
