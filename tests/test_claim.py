"""Schema tests: the claim/verdict contracts every pipeline stage depends on."""

import pytest
from pydantic import ValidationError

from anchor import Claim, Evidence, Verdict, VerdictLabel, VerifiedClaim


def make_claim() -> Claim:
    return Claim(
        text="FY2025 revenue was $4.2B",
        source_text="revenue grew 12% to $4.2B in FY2025",
    )


def make_evidence() -> Evidence:
    return Evidence(chunk_id="10k-2025-item7-003", quote="Total revenue was $4.2 billion")


class TestClaim:
    def test_constructs(self) -> None:
        claim = make_claim()
        assert claim.text == "FY2025 revenue was $4.2B"
        assert claim.source_text == "revenue grew 12% to $4.2B in FY2025"

    @pytest.mark.parametrize("field", ["text", "source_text"])
    def test_rejects_empty_fields(self, field: str) -> None:
        data = {"text": "a claim", "source_text": "a sentence", field: ""}
        with pytest.raises(ValidationError):
            Claim(**data)

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            make_claim().text = "changed"  # type: ignore[misc]


class TestVerdict:
    def test_vocabulary_is_fixed(self) -> None:
        with pytest.raises(ValidationError):
            Verdict(label="plausible", rationale="not a real verdict")  # type: ignore[arg-type]

    def test_supported_requires_evidence(self) -> None:
        with pytest.raises(ValidationError, match="must cite evidence"):
            Verdict(label=VerdictLabel.SUPPORTED, rationale="the context says so")

    def test_contradicted_requires_evidence(self) -> None:
        with pytest.raises(ValidationError, match="must cite evidence"):
            Verdict(label=VerdictLabel.CONTRADICTED, rationale="the context disagrees")

    def test_unsupported_forbids_evidence(self) -> None:
        with pytest.raises(ValidationError, match="cannot cite evidence"):
            Verdict(
                label=VerdictLabel.UNSUPPORTED,
                rationale="nothing in context",
                evidence=make_evidence(),
            )

    def test_unsupported_without_evidence_is_valid(self) -> None:
        verdict = Verdict(label=VerdictLabel.UNSUPPORTED, rationale="no span mentions revenue")
        assert verdict.evidence is None

    def test_supported_with_evidence_is_valid(self) -> None:
        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            rationale="chunk states the figure",
            evidence=make_evidence(),
        )
        assert verdict.evidence is not None
        assert verdict.evidence.chunk_id == "10k-2025-item7-003"


class TestVerifiedClaim:
    def test_pairs_claim_with_verdict(self) -> None:
        verified = VerifiedClaim(
            claim=make_claim(),
            verdict=Verdict(
                label=VerdictLabel.SUPPORTED,
                rationale="chunk states the figure",
                evidence=make_evidence(),
            ),
        )
        assert verified.verdict.label is VerdictLabel.SUPPORTED

    def test_round_trips_through_json(self) -> None:
        verified = VerifiedClaim(
            claim=make_claim(),
            verdict=Verdict(
                label=VerdictLabel.CONTRADICTED,
                rationale="context reports a different figure",
                evidence=make_evidence(),
            ),
        )
        restored = VerifiedClaim.model_validate_json(verified.model_dump_json())
        assert restored == verified
