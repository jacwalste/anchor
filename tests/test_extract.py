"""Extraction tests: parsing, validation, and provenance-checking of judge output.

These prove plumbing and parsing — NOT extraction quality, which is a property
of the live judge model (see PLAN.md §3).
"""

import json

import pytest

from anchor import Claim, JudgeResponseError, extract_claims
from tests.stubs import ScriptedJudge

ANSWER = "Acme's revenue grew 12% to $4.2B in FY2025."

VALID_RESPONSE = json.dumps(
    [
        {"text": "Acme's revenue grew 12% in FY2025", "source_text": "revenue grew 12%"},
        {"text": "Acme's FY2025 revenue was $4.2B", "source_text": "to $4.2B in FY2025"},
    ]
)


def test_extracts_claims_from_valid_response() -> None:
    judge = ScriptedJudge(responses=[VALID_RESPONSE])
    claims = extract_claims(ANSWER, judge)
    assert claims == [
        Claim(text="Acme's revenue grew 12% in FY2025", source_text="revenue grew 12%"),
        Claim(text="Acme's FY2025 revenue was $4.2B", source_text="to $4.2B in FY2025"),
    ]


def test_prompt_contains_the_answer() -> None:
    judge = ScriptedJudge(responses=[VALID_RESPONSE])
    extract_claims(ANSWER, judge)
    assert len(judge.prompts) == 1
    assert ANSWER in judge.prompts[0]


def test_handles_markdown_fenced_json() -> None:
    judge = ScriptedJudge(responses=[f"```json\n{VALID_RESPONSE}\n```"])
    claims = extract_claims(ANSWER, judge)
    assert len(claims) == 2


def test_rejects_invalid_json() -> None:
    judge = ScriptedJudge(responses=["Sure! Here are the claims: revenue grew"])
    with pytest.raises(JudgeResponseError, match="not valid JSON"):
        extract_claims(ANSWER, judge)


def test_rejects_non_array_json() -> None:
    judge = ScriptedJudge(responses=['{"claims": []}'])
    with pytest.raises(JudgeResponseError, match="expected a JSON array"):
        extract_claims(ANSWER, judge)


def test_rejects_items_missing_fields() -> None:
    judge = ScriptedJudge(responses=['[{"text": "a claim with no source_text"}]'])
    with pytest.raises(JudgeResponseError, match="schema validation"):
        extract_claims(ANSWER, judge)


def test_rejects_source_text_absent_from_answer() -> None:
    fabricated = json.dumps(
        [{"text": "Acme's margin improved", "source_text": "margins improved sharply"}]
    )
    judge = ScriptedJudge(responses=[fabricated])
    with pytest.raises(JudgeResponseError, match="not found verbatim"):
        extract_claims(ANSWER, judge)


def test_error_carries_raw_response() -> None:
    raw = "not json at all"
    judge = ScriptedJudge(responses=[raw])
    with pytest.raises(JudgeResponseError) as exc_info:
        extract_claims(ANSWER, judge)
    assert exc_info.value.raw_response == raw
    assert raw in str(exc_info.value)


def test_empty_claim_list_is_returned_not_raised() -> None:
    # Zero claims is legal here ("I don't know" has nothing verifiable);
    # the SCORER is what refuses to score an empty list (D10).
    judge = ScriptedJudge(responses=["[]"])
    assert extract_claims("I could not find that information.", judge) == []


def test_rejects_empty_answer_without_calling_judge() -> None:
    judge = ScriptedJudge(responses=[])
    with pytest.raises(ValueError, match="empty answer"):
        extract_claims("   ", judge)
    assert judge.prompts == []
