"""Test doubles and factories shared across the suite. No real model calls, ever."""

from anchor import Claim, Evidence, Verdict, VerdictLabel, VerifiedClaim


def make_verified_claim(
    label: VerdictLabel,
    claim_text: str = "a canned claim",
    quote: str = "a canned supporting passage",
) -> VerifiedClaim:
    """Build a schema-valid VerifiedClaim: evidence present unless unsupported."""
    if label is VerdictLabel.UNSUPPORTED:
        evidence = None
    else:
        evidence = Evidence(chunk_id="chunk-1", quote=quote)
    return VerifiedClaim(
        claim=Claim(text=claim_text, source_text=claim_text),
        verdict=Verdict(label=label, rationale="canned rationale", evidence=evidence),
    )


class ScriptedJudge:
    """A JudgeClient that replays canned responses in order.

    Records every prompt it receives so tests can assert on prompt content.
    Fails loudly if asked for more responses than it was given — a test that
    over-calls the judge is a test with a bug.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError(
                f"ScriptedJudge exhausted after {len(self.prompts) - 1} responses"
            )
        return self._responses.pop(0)
