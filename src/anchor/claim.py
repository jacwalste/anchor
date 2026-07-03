"""Core data contracts: claims, evidence, verdicts.

This is the load-bearing seam of the harness — extract, verify, score, and
report all communicate through these types. Changes here require a test and a
changelog note (CLAUDE.md guardrails).
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerdictLabel(StrEnum):
    """Fixed verdict vocabulary — do not expand in v1."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class Claim(BaseModel):
    """One atomic, independently verifiable statement extracted from an answer.

    `source_text` is the verbatim answer snippet the claim was derived from —
    positions are derived by code when needed, never produced by the judge.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    source_text: str = Field(min_length=1)


class Evidence(BaseModel):
    """A context span backing a verdict: chunk reference plus verbatim quote.

    The quote comes from the judge and must be located in the actual chunk by
    the verifier before it is trusted (quote-then-locate).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Verdict(BaseModel):
    """The verifier's judgment of one claim against the retrieved context.

    Evidence is required for supported/contradicted (the span that backs the
    judgment) and forbidden for unsupported (unsupported MEANS no span exists).
    """

    model_config = ConfigDict(frozen=True)

    label: VerdictLabel
    rationale: str = Field(min_length=1)
    evidence: Evidence | None = None

    @model_validator(mode="after")
    def _evidence_must_match_label(self) -> Self:
        if self.label is VerdictLabel.UNSUPPORTED:
            if self.evidence is not None:
                raise ValueError("an unsupported verdict cannot cite evidence")
        elif self.evidence is None:
            raise ValueError(f"a {self.label.value} verdict must cite evidence")
        return self


class VerifiedClaim(BaseModel):
    """A claim paired with its verdict — the unit the scorer and reporter consume."""

    model_config = ConfigDict(frozen=True)

    claim: Claim
    verdict: Verdict
