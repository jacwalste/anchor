"""Benchmark runner: evaluate labeled cases and check planted hallucinations.

Matching criterion (PLAN.md §6.7): the extractor decomposes answers
differently than the labels, so a planted hallucination counts as CAUGHT iff
at least one extracted claim whose source_text overlaps the planted span got
a non-supported verdict. Overlap is computed on first occurrences within the
answer — both spans are validated verbatim substrings of it.

A judge failure on one case is recorded on that case and the run continues;
losing one answer to a judge hiccup must not lose the whole benchmark.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field

from anchor.claim import Chunk, VerdictLabel, VerifiedClaim
from anchor.dataset import BenchmarkCase, PlantedHallucination, resolve_context
from anchor.errors import JudgeResponseError
from anchor.evaluate import evaluate_answer
from anchor.judge import JudgeClient
from anchor.score import GroundednessScore, score_claims


class PlantedOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    planted: PlantedHallucination
    caught: bool


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    verified: list[VerifiedClaim] = Field(default_factory=list)
    planted_outcomes: list[PlantedOutcome] = Field(default_factory=list)
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> GroundednessScore | None:
        if self.error is not None or not self.verified:
            return None
        return score_claims(self.verified)


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cases: list[CaseResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def planted_count(self) -> int:
        return sum(len(case.planted_outcomes) for case in self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def caught_count(self) -> int:
        return sum(1 for case in self.cases for outcome in case.planted_outcomes if outcome.caught)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errored_case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases if case.error is not None]


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    corpus: Sequence[Chunk],
    judge: JudgeClient,
    *,
    max_concurrency: int = 1,
) -> BenchmarkResult:
    results: list[CaseResult] = []
    for case in cases:
        context = resolve_context(case, corpus)
        try:
            verified = evaluate_answer(case.answer, context, judge, max_concurrency=max_concurrency)
        except JudgeResponseError as exc:
            results.append(CaseResult(case_id=case.id, error=str(exc)))
            continue
        outcomes = [
            PlantedOutcome(planted=planted, caught=_is_caught(case.answer, planted, verified))
            for planted in case.planted
        ]
        results.append(CaseResult(case_id=case.id, verified=verified, planted_outcomes=outcomes))
    return BenchmarkResult(cases=results)


def _is_caught(
    answer: str, planted: PlantedHallucination, verified: Sequence[VerifiedClaim]
) -> bool:
    return any(
        item.verdict.label is not VerdictLabel.SUPPORTED
        and _overlaps(answer, planted.span, item.claim.source_text)
        for item in verified
    )


def _overlaps(answer: str, span_a: str, span_b: str) -> bool:
    start_a, start_b = answer.find(span_a), answer.find(span_b)
    return start_a < start_b + len(span_b) and start_b < start_a + len(span_a)
