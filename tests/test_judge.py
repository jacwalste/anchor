"""The judge client seam: protocol shape and the test stub's own behavior."""

import pytest

from anchor import JudgeClient
from tests.stubs import ScriptedJudge


def test_scripted_judge_satisfies_protocol() -> None:
    assert isinstance(ScriptedJudge(responses=[]), JudgeClient)


def test_replays_responses_in_order() -> None:
    judge = ScriptedJudge(responses=["first", "second"])
    assert judge.complete("prompt a") == "first"
    assert judge.complete("prompt b") == "second"


def test_records_prompts() -> None:
    judge = ScriptedJudge(responses=["ok"])
    judge.complete("the prompt under test")
    assert judge.prompts == ["the prompt under test"]


def test_fails_loudly_when_exhausted() -> None:
    judge = ScriptedJudge(responses=["only one"])
    judge.complete("first call")
    with pytest.raises(AssertionError, match="exhausted"):
        judge.complete("one call too many")
