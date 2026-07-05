"""Claim extraction: decompose an answer into atomic claims via the judge.

This stage owns the extraction prompt and all parsing/validation of the raw
judge response — the judge client itself is just prompt-in, text-out.
"""

from pydantic import ValidationError

from anchor.claim import Claim
from anchor.errors import JudgeResponseError
from anchor.judge import JudgeClient
from anchor.parsing import parse_json

_EXTRACTION_PROMPT = """\
Decompose the answer below into atomic claims. An atomic claim states exactly \
one independently verifiable fact; a sentence asserting several facts must be \
split into several claims.

Respond with ONLY a JSON array, one object per claim:
[{{"text": "<the claim, worded as a standalone statement>", \
"source_text": "<the answer snippet it came from>"}}]

Rules:
- "source_text" must be copied character-for-character from the answer.
- Do not include opinions, hedges, or questions — only verifiable factual claims.
- If the answer contains no verifiable claims, respond with [].

Answer:
{answer}
"""


def extract_claims(answer: str, judge: JudgeClient) -> list[Claim]:
    """Decompose `answer` into atomic claims, validating everything the judge returns.

    Raises JudgeResponseError if the response is not a JSON array of valid
    claims, or if any claim's source_text does not appear verbatim in the
    answer (a claim whose provenance can't be confirmed is not usable).
    """
    if not answer.strip():
        raise ValueError("cannot extract claims from an empty answer")

    raw = judge.complete(_EXTRACTION_PROMPT.format(answer=answer))
    claims: list[Claim] = []
    for item in _parse_json_array(raw):
        try:
            claim = Claim.model_validate(item)
        except ValidationError as exc:
            raise JudgeResponseError(f"claim failed schema validation: {exc}", raw) from exc
        if claim.source_text not in answer:
            raise JudgeResponseError(
                f"source_text not found verbatim in the answer: {claim.source_text!r}", raw
            )
        claims.append(claim)
    return claims


def _parse_json_array(raw: str) -> list[object]:
    parsed = parse_json(raw)
    if not isinstance(parsed, list):
        raise JudgeResponseError(f"expected a JSON array, got {type(parsed).__name__}", raw)
    return parsed
