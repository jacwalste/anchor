"""Exceptions for untrustworthy judge output."""


class JudgeResponseError(Exception):
    """A judge response failed parsing or validation.

    Always carries the raw response — a failure you can't inspect is a
    failure you can't debug. No silent retries in v1: this raises to the
    caller (PLAN.md D4 notes).
    """

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(f"{message}\n--- raw judge response ---\n{raw_response}")
        self.raw_response = raw_response
