"""The judge client seam — the only place the harness touches a model.

The contract is deliberately minimal: prompt in, raw text out. Prompt
construction, parsing, validation, and retry policy all live in the harness
(extract/verify), uniformly across judges. Implementations may use provider
features internally (structured-output modes, tool calling) — the harness
validates every response regardless and never depends on them.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgeClient(Protocol):
    """Minimal completion client backing claim extraction and verification."""

    def complete(self, prompt: str) -> str:
        """Return the model's raw text response to `prompt`."""
        ...
