"""The harness front door: evaluate one (answer, retrieved-context) pair."""

from collections.abc import Sequence

from anchor.claim import Chunk, VerifiedClaim
from anchor.extract import extract_claims
from anchor.judge import JudgeClient
from anchor.verify import verify_claim


def evaluate_answer(
    answer: str, context: Sequence[Chunk], judge: JudgeClient
) -> list[VerifiedClaim]:
    """Extract atomic claims from `answer` and verify each against `context`.

    Feed the result to `score_claims` for the score or `render_markdown`
    for the report.
    """
    claims = extract_claims(answer, judge)
    return [
        VerifiedClaim(claim=claim, verdict=verify_claim(claim, context, judge))
        for claim in claims
    ]
