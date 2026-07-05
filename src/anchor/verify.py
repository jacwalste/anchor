"""Per-claim verification: judge one claim against the retrieved context.

This stage owns the verification prompt — including the context-only rule
(D3) — and all parsing/validation of the raw judge response. Evidence cited
by the judge is never trusted: the chunk id must resolve to a real chunk and
the quote must appear verbatim in it (quote-then-locate, D8).
"""

from collections.abc import Sequence

from pydantic import ValidationError

from anchor.claim import Chunk, Claim, Verdict
from anchor.errors import JudgeResponseError
from anchor.judge import JudgeClient
from anchor.parsing import parse_json

_VERIFICATION_PROMPT = """\
Judge whether the claim below is supported by the retrieved context. The \
context is the ONLY admissible evidence: a claim that is true in the real \
world but not stated in the context is "unsupported".

Verdicts:
- "supported": a context passage entails the claim.
- "unsupported": no context passage supports the claim.
- "contradicted": a context passage states the opposite of the claim.

Respond with ONLY a JSON object:
{{"label": "<supported|unsupported|contradicted>", \
"rationale": "<one or two sentences>", \
"evidence": {{"chunk_id": "<id>", "quote": "<passage copied VERBATIM from that chunk>"}}}}

For "unsupported", set "evidence" to null. For "supported" and "contradicted", \
"evidence" is required and the quote must be copied character-for-character \
from the cited chunk.

Claim:
{claim}

Context:
{context}
"""


def verify_claim(claim: Claim, context: Sequence[Chunk], judge: JudgeClient) -> Verdict:
    """Judge `claim` against `context`, validating everything the judge returns.

    Raises JudgeResponseError if the response is not a valid verdict, cites a
    chunk that wasn't provided, or quotes text that doesn't appear in the
    cited chunk.
    """
    if not context:
        raise ValueError("cannot verify a claim against empty context")

    prompt = _VERIFICATION_PROMPT.format(claim=claim.text, context=_format_context(context))
    raw = judge.complete(prompt)

    parsed = parse_json(raw)
    if not isinstance(parsed, dict):
        raise JudgeResponseError(f"expected a JSON object, got {type(parsed).__name__}", raw)
    try:
        verdict = Verdict.model_validate(parsed)
    except ValidationError as exc:
        raise JudgeResponseError(f"verdict failed schema validation: {exc}", raw) from exc

    if verdict.evidence is not None:
        cited = next((c for c in context if c.id == verdict.evidence.chunk_id), None)
        if cited is None:
            raise JudgeResponseError(
                f"evidence cites unknown chunk id: {verdict.evidence.chunk_id!r}", raw
            )
        if verdict.evidence.quote not in cited.text:
            raise JudgeResponseError(
                f"evidence quote not found verbatim in chunk {cited.id!r}: "
                f"{verdict.evidence.quote!r}",
                raw,
            )
    return verdict


def _format_context(context: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in context)
