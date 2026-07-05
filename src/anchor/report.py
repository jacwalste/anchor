"""Reporting: render verified claims as a claim-level markdown report.

Flagged (unsupported/contradicted) claims lead the report — localization is
the product; the scalar is just the summary line.
"""

from collections.abc import Sequence

from anchor.claim import VerdictLabel, VerifiedClaim
from anchor.score import score_claims


def render_markdown(verified: Sequence[VerifiedClaim]) -> str:
    """Render the claim-level groundedness report.

    Computes the score from `verified` itself, so the report can never
    disagree with the claims it displays.
    """
    result = score_claims(verified)
    lines = [
        "# Groundedness Report",
        "",
        f"**Score: {result.score:.2f}** — {result.supported} of {result.total} claims supported",
        "",
        f"Breakdown: {result.supported} supported / {result.unsupported} unsupported / "
        f"{result.contradicted} contradicted",
        "",
        "## Flagged claims",
        "",
    ]

    flagged = [v for v in verified if v.verdict.label is not VerdictLabel.SUPPORTED]
    if flagged:
        for item in flagged:
            lines.extend(_render_claim(item))
    else:
        lines.extend(["None — every claim is supported by the retrieved context.", ""])

    lines.extend(["## All claims", ""])
    for number, item in enumerate(verified, start=1):
        lines.extend(_render_claim(item, number=number))

    return "\n".join(lines)


def _render_claim(item: VerifiedClaim, number: int | None = None) -> list[str]:
    prefix = f"{number}. " if number is not None else "- "
    indent = " " * len(prefix)
    lines = [
        f"{prefix}**{item.verdict.label.value}** — {item.claim.text}",
        f'{indent}- from answer: "{item.claim.source_text}"',
    ]
    if item.verdict.evidence is not None:
        lines.append(
            f'{indent}- evidence [{item.verdict.evidence.chunk_id}]: '
            f'"{item.verdict.evidence.quote}"'
        )
    lines.extend([f"{indent}- rationale: {item.verdict.rationale}", ""])
    return lines
