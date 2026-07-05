"""Scoring: aggregate per-claim verdicts into a groundedness score."""

from collections import Counter
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from anchor.claim import VerdictLabel, VerifiedClaim


class GroundednessScore(BaseModel):
    """Verdict counts plus derived totals.

    Only the three counts are stored; total/score/hallucinated are computed
    from them, so the numbers can never disagree with each other. A zero
    total is unrepresentable: scoring nothing means extraction broke
    upstream (D10).
    """

    model_config = ConfigDict(frozen=True)

    supported: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    contradicted: int = Field(ge=0)

    @model_validator(mode="after")
    def _reject_zero_claims(self) -> Self:
        if self.total == 0:
            raise ValueError("cannot score zero claims — empty extraction is an upstream failure")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return self.supported + self.unsupported + self.contradicted

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float:
        """Groundedness: fraction of claims supported by the retrieved context."""
        return self.supported / self.total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hallucinated(self) -> int:
        """Claims not grounded in the context: unsupported + contradicted."""
        return self.unsupported + self.contradicted


def score_claims(verified: Sequence[VerifiedClaim]) -> GroundednessScore:
    """Aggregate verdicts into a GroundednessScore. Raises on an empty list (D10)."""
    counts = Counter(v.verdict.label for v in verified)
    return GroundednessScore(
        supported=counts[VerdictLabel.SUPPORTED],
        unsupported=counts[VerdictLabel.UNSUPPORTED],
        contradicted=counts[VerdictLabel.CONTRADICTED],
    )
