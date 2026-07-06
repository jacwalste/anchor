"""Reporting: render verified claims as a claim-level markdown report.

Flagged (unsupported/contradicted) claims lead the report — localization is
the product; the scalar is just the summary line.
"""

from collections.abc import Sequence

from anchor.benchmark import BenchmarkResult
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


def render_benchmark_markdown(result: BenchmarkResult) -> str:
    """Render a benchmark run: catch rate up top, per-case detail below."""
    lines = [
        "# Anchor Benchmark Report",
        "",
        f"Cases: {len(result.cases)} · Planted hallucinations caught: "
        f"**{result.caught_count} of {result.planted_count}**",
    ]
    if result.errored_case_ids:
        lines.append(f"Cases with judge errors: {', '.join(result.errored_case_ids)}")
    lines.append("")

    for case in result.cases:
        lines.extend([f"## {case.case_id}", ""])
        if case.error is not None:
            lines.extend([f"JUDGE ERROR: {case.error}", ""])
            continue
        score = case.score
        if score is None:
            lines.extend(["No claims extracted.", ""])
            continue
        lines.extend(
            [f"Score: **{score.score:.2f}** ({score.supported} of {score.total} supported)", ""]
        )
        for outcome in case.planted_outcomes:
            status = "CAUGHT" if outcome.caught else "MISSED"
            lines.append(
                f'- {status} (expected {outcome.planted.expected.value}): "{outcome.planted.span}"'
            )
        if case.planted_outcomes:
            lines.append("")
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
