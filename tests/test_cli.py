"""CLI tests: argument wiring and output, judge stubbed via _make_judge."""

import json
from pathlib import Path

import pytest

from anchor import cli
from tests.stubs import ScriptedJudge


@pytest.fixture()
def data_files(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"id": "c1", "text": "Revenue was $4.2 billion."}]))
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "question": "What was revenue?",
                    "answer": "Revenue was $4.2 billion. AcmeCloud leads the market.",
                    "chunk_ids": ["c1"],
                    "planted": [{"span": "AcmeCloud leads the market.", "expected": "unsupported"}],
                }
            ]
        )
    )
    return corpus, cases


def test_benchmark_command_prints_report_and_exits_zero(
    data_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, cases = data_files
    judge = ScriptedJudge(
        responses=[
            json.dumps(
                [
                    {"text": "Revenue was $4.2B", "source_text": "Revenue was $4.2 billion"},
                    {"text": "AcmeCloud leads the market", "source_text": "leads the market"},
                ]
            ),
            json.dumps(
                {
                    "label": "supported",
                    "rationale": "Stated directly.",
                    "evidence": {"chunk_id": "c1", "quote": "$4.2 billion"},
                }
            ),
            json.dumps({"label": "unsupported", "rationale": "Not in context.", "evidence": None}),
        ]
    )
    monkeypatch.setattr(cli, "_make_judge", lambda model: judge)

    # ScriptedJudge replays responses in order, so pin verification to sequential.
    exit_code = cli.main(
        ["benchmark", "--corpus", str(corpus), "--cases", str(cases), "--judge-model", "stub"]
        + ["--max-concurrency", "1"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1 of 1" in out
    assert "CAUGHT" in out


def test_benchmark_command_emits_machine_readable_json(
    data_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, cases = data_files
    judge = ScriptedJudge(
        responses=[
            json.dumps(
                [
                    {"text": "Revenue was $4.2B", "source_text": "Revenue was $4.2 billion"},
                    {"text": "AcmeCloud leads the market", "source_text": "leads the market"},
                ]
            ),
            json.dumps(
                {
                    "label": "supported",
                    "rationale": "Stated directly.",
                    "evidence": {"chunk_id": "c1", "quote": "$4.2 billion"},
                }
            ),
            json.dumps({"label": "unsupported", "rationale": "Not in context.", "evidence": None}),
        ]
    )
    monkeypatch.setattr(cli, "_make_judge", lambda model: judge)

    exit_code = cli.main(
        ["benchmark", "--corpus", str(corpus), "--cases", str(cases), "--judge-model", "stub"]
        + ["--max-concurrency", "1", "--format", "json"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["caught_count"] == 1
    assert payload["planted_count"] == 1
    assert payload["cases"][0]["score"]["score"] == 0.5


def test_ask_command_emits_json_with_answer_and_score(
    data_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, _ = data_files
    agent_model = ScriptedJudge(
        responses=[
            json.dumps({"sufficient": True, "refined_query": None}),
            "Revenue was $4.2 billion [c1].",
        ]
    )
    judge = ScriptedJudge(
        responses=[
            json.dumps([{"text": "Revenue was $4.2B", "source_text": "Revenue was $4.2 billion"}]),
            json.dumps(
                {
                    "label": "supported",
                    "rationale": "Stated directly.",
                    "evidence": {"chunk_id": "c1", "quote": "$4.2 billion"},
                }
            ),
        ]
    )
    judges = {"agent-m": agent_model, "judge-m": judge}
    monkeypatch.setattr(cli, "_make_judge", lambda model: judges[model])

    exit_code = cli.main(
        ["ask", "--corpus", str(corpus), "--question", "What was revenue?"]
        + ["--agent-model", "agent-m", "--judge-model", "judge-m"]
        + ["--max-concurrency", "1", "--format", "json"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "Revenue was $4.2 billion [c1]."
    assert payload["score"]["score"] == 1.0
    assert payload["claims"][0]["verdict"]["label"] == "supported"


def test_benchmark_command_exits_nonzero_on_judge_errors(
    data_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, cases = data_files
    monkeypatch.setattr(cli, "_make_judge", lambda model: ScriptedJudge(responses=["garbage"]))

    exit_code = cli.main(
        ["benchmark", "--corpus", str(corpus), "--cases", str(cases), "--judge-model", "stub"]
        + ["--max-concurrency", "1"]
    )

    assert exit_code == 1
    assert "JUDGE ERROR" in capsys.readouterr().out
