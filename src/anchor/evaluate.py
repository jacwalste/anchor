"""The harness front door: evaluate one (answer, retrieved-context) pair."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from anchor.claim import Chunk, VerifiedClaim
from anchor.extract import extract_claims
from anchor.judge import JudgeClient
from anchor.verify import verify_claim


def evaluate_answer(
    answer: str,
    context: Sequence[Chunk],
    judge: JudgeClient,
    *,
    max_concurrency: int = 1,
) -> list[VerifiedClaim]:
    """Extract atomic claims from `answer` and verify each against `context`.

    Claims verify independently, so with max_concurrency > 1 the verification
    calls run on a thread pool — the judge must be thread-safe (the shipped
    AnthropicJudge is). Results keep claim order either way, and the first
    verification failure propagates: no partial results.

    Feed the result to `score_claims` for the score or `render_markdown`
    for the report.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    claims = extract_claims(answer, judge)
    if max_concurrency == 1 or len(claims) <= 1:
        verdicts = [verify_claim(claim, context, judge) for claim in claims]
    else:
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(claims))) as pool:
            verdicts = list(pool.map(lambda claim: verify_claim(claim, context, judge), claims))
    return [
        VerifiedClaim(claim=claim, verdict=verdict)
        for claim, verdict in zip(claims, verdicts, strict=True)
    ]
