"""Report tests: the claim-level markdown report renders what it claims to."""

import pytest

from anchor import VerdictLabel, render_markdown
from tests.stubs import make_verified_claim

SUPPORTED = make_verified_claim(
    VerdictLabel.SUPPORTED,
    claim_text="Revenue was $4.2B in FY2025",
    quote="Total revenue for fiscal 2025 was $4.2 billion.",
)
UNSUPPORTED = make_verified_claim(
    VerdictLabel.UNSUPPORTED,
    claim_text="Acme leads the market in cloud services",
)


def test_report_shows_score_and_breakdown() -> None:
    report = render_markdown([SUPPORTED, UNSUPPORTED])
    assert "0.50" in report
    assert "1 of 2 claims supported" in report
    assert "1 supported" in report
    assert "1 unsupported" in report


def test_report_shows_every_claim_with_verdict_and_evidence() -> None:
    report = render_markdown([SUPPORTED, UNSUPPORTED])
    assert "Revenue was $4.2B in FY2025" in report
    assert "Acme leads the market in cloud services" in report
    assert "supported" in report
    assert "unsupported" in report
    assert "Total revenue for fiscal 2025 was $4.2 billion." in report
    assert "[chunk-1]" in report


def test_flagged_section_contains_only_ungrounded_claims() -> None:
    report = render_markdown([SUPPORTED, UNSUPPORTED])
    flagged_section = report.split("## Flagged claims")[1].split("## All claims")[0]
    assert "Acme leads the market in cloud services" in flagged_section
    assert "Revenue was $4.2B in FY2025" not in flagged_section


def test_fully_grounded_answer_flags_nothing() -> None:
    report = render_markdown([SUPPORTED])
    flagged_section = report.split("## Flagged claims")[1].split("## All claims")[0]
    assert "None" in flagged_section


def test_zero_claims_raises() -> None:
    with pytest.raises(ValueError, match="zero claims"):
        render_markdown([])
