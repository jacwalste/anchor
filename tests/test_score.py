"""Scoring tests: verdict counts to groundedness score, D10 zero-claims rule."""

import pytest
from pydantic import ValidationError

from anchor import GroundednessScore, VerdictLabel, score_claims
from tests.stubs import make_verified_claim


def test_scores_known_verdicts() -> None:
    verified = [
        make_verified_claim(VerdictLabel.SUPPORTED),
        make_verified_claim(VerdictLabel.SUPPORTED),
        make_verified_claim(VerdictLabel.UNSUPPORTED),
    ]
    result = score_claims(verified)
    assert result.supported == 2
    assert result.unsupported == 1
    assert result.contradicted == 0
    assert result.total == 3
    assert result.score == pytest.approx(2 / 3)


def test_perfectly_grounded_answer_scores_one() -> None:
    verified = [make_verified_claim(VerdictLabel.SUPPORTED)]
    assert score_claims(verified).score == 1.0


def test_contradicted_counts_against_the_score() -> None:
    verified = [
        make_verified_claim(VerdictLabel.SUPPORTED),
        make_verified_claim(VerdictLabel.CONTRADICTED),
    ]
    result = score_claims(verified)
    assert result.score == pytest.approx(0.5)
    assert result.hallucinated == 1


def test_zero_claims_raises() -> None:
    with pytest.raises(ValueError, match="zero claims"):
        score_claims([])


def test_zero_total_is_unrepresentable_even_by_direct_construction() -> None:
    with pytest.raises(ValidationError, match="zero claims"):
        GroundednessScore(supported=0, unsupported=0, contradicted=0)


def test_serializes_with_derived_fields() -> None:
    result = score_claims(
        [
            make_verified_claim(VerdictLabel.SUPPORTED),
            make_verified_claim(VerdictLabel.UNSUPPORTED),
        ]
    )
    dumped = result.model_dump()
    assert dumped["total"] == 2
    assert dumped["score"] == pytest.approx(0.5)
    assert dumped["hallucinated"] == 1
