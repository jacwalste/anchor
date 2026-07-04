"""Test doubles shared across the suite. No real model calls, ever."""


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
