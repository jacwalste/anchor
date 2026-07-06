"""AnthropicJudge tests with a fake client object — no network, ever."""

from types import SimpleNamespace
from typing import Any

import pytest

from anchor import JudgeClient
from anchor.judge_anthropic import AnthropicJudge


class FakeClient:
    def __init__(self, blocks: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._blocks = blocks
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._blocks)


def text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def test_satisfies_judge_protocol() -> None:
    judge = AnthropicJudge(model="stub-model", client=FakeClient([text_block("ok")]))  # type: ignore[arg-type]
    assert isinstance(judge, JudgeClient)


def test_passes_model_and_prompt_and_returns_text() -> None:
    fake = FakeClient([text_block("the response")])
    judge = AnthropicJudge(model="stub-model", client=fake)  # type: ignore[arg-type]
    assert judge.complete("the prompt") == "the response"
    (call,) = fake.calls
    assert call["model"] == "stub-model"
    assert call["messages"] == [{"role": "user", "content": "the prompt"}]


def test_no_text_blocks_fails_loudly() -> None:
    fake = FakeClient([SimpleNamespace(type="thinking", thinking="hmm")])
    judge = AnthropicJudge(model="stub-model", client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no text blocks"):
        judge.complete("the prompt")
