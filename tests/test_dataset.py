"""Dataset tests: loader validation rules, and integrity of the SHIPPED data files."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from anchor import (
    BenchmarkCase,
    PlantedHallucination,
    VerdictLabel,
    load_benchmark,
    load_chunks,
    resolve_context,
)

DATA = Path(__file__).parent.parent / "data"
CORPUS_PATH = DATA / "corpus" / "acmecloud_fy2025.json"
BENCHMARK_PATH = DATA / "benchmark" / "cases.json"


class TestShippedData:
    """The repo's own data files must pass their loaders — authoring bugs fail CI."""

    def test_corpus_loads(self) -> None:
        chunks = load_chunks(CORPUS_PATH)
        assert len(chunks) >= 10

    def test_benchmark_loads_and_has_planted_hallucinations(self) -> None:
        cases = load_benchmark(BENCHMARK_PATH)
        assert len(cases) >= 5
        assert any(case.planted for case in cases)
        assert any(not case.planted for case in cases)  # a fully grounded control

    def test_every_case_resolves_against_the_corpus(self) -> None:
        corpus = load_chunks(CORPUS_PATH)
        for case in load_benchmark(BENCHMARK_PATH):
            context = resolve_context(case, corpus)
            assert len(context) == len(case.chunk_ids)

    def test_benchmark_covers_both_hallucination_kinds(self) -> None:
        planted = [p for case in load_benchmark(BENCHMARK_PATH) for p in case.planted]
        kinds = {p.expected for p in planted}
        assert kinds == {VerdictLabel.UNSUPPORTED, VerdictLabel.CONTRADICTED}


class TestLoaderValidation:
    def test_duplicate_chunk_ids_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.json"
        path.write_text('[{"id": "c1", "text": "a"}, {"id": "c1", "text": "b"}]')
        with pytest.raises(ValueError, match="duplicate chunk ids"):
            load_chunks(path)

    def test_planted_span_must_be_verbatim_in_answer(self) -> None:
        with pytest.raises(ValidationError, match="not found verbatim"):
            BenchmarkCase(
                id="case-x",
                question="q?",
                answer="The margin was 18%.",
                chunk_ids=["c1"],
                planted=[
                    PlantedHallucination(span="margin was 21%", expected=VerdictLabel.UNSUPPORTED)
                ],
            )

    def test_planted_hallucination_cannot_expect_supported(self) -> None:
        with pytest.raises(ValidationError, match="cannot have expected verdict"):
            PlantedHallucination(span="anything", expected=VerdictLabel.SUPPORTED)

    def test_unknown_chunk_ids_rejected_at_resolution(self) -> None:
        case = BenchmarkCase(
            id="case-x", question="q?", answer="a.", chunk_ids=["nope"], planted=[]
        )
        with pytest.raises(ValueError, match="unknown chunk ids"):
            resolve_context(case, load_chunks(CORPUS_PATH))
