"""Loaders for corpus and benchmark data files.

Validation is strict and loud at load time: a duplicate chunk id, a planted
span that isn't verbatim in its answer, or a case citing an unknown chunk is
an authoring bug — better a crash here than a silently meaningless benchmark.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from anchor.claim import Chunk, VerdictLabel


class PlantedHallucination(BaseModel):
    """A known-bad span deliberately written into a benchmark answer."""

    model_config = ConfigDict(frozen=True)

    span: str = Field(min_length=1)
    expected: VerdictLabel

    @field_validator("expected")
    @classmethod
    def _must_be_a_hallucination(cls, value: VerdictLabel) -> VerdictLabel:
        if value is VerdictLabel.SUPPORTED:
            raise ValueError("a planted hallucination cannot have expected verdict 'supported'")
        return value


class BenchmarkCase(BaseModel):
    """A labeled answer: fixed text, the chunks 'retrieved' for it, planted spans."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)
    planted: list[PlantedHallucination] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_must_appear_verbatim(self) -> Self:
        for planted in self.planted:
            if planted.span not in self.answer:
                raise ValueError(
                    f"planted span not found verbatim in answer of case {self.id!r}: "
                    f"{planted.span!r}"
                )
        return self


_CHUNKS = TypeAdapter(list[Chunk])
_CASES = TypeAdapter(list[BenchmarkCase])


def load_chunks(path: Path) -> list[Chunk]:
    chunks = _CHUNKS.validate_json(path.read_text())
    _ensure_unique_ids([chunk.id for chunk in chunks], what="chunk", path=path)
    return chunks


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    cases = _CASES.validate_json(path.read_text())
    _ensure_unique_ids([case.id for case in cases], what="benchmark case", path=path)
    return cases


def resolve_context(case: BenchmarkCase, corpus: Sequence[Chunk]) -> list[Chunk]:
    """Turn a case's chunk_ids into Chunks, refusing ids the corpus doesn't have."""
    by_id = {chunk.id: chunk for chunk in corpus}
    missing = [chunk_id for chunk_id in case.chunk_ids if chunk_id not in by_id]
    if missing:
        raise ValueError(f"benchmark case {case.id!r} references unknown chunk ids: {missing}")
    return [by_id[chunk_id] for chunk_id in case.chunk_ids]


def _ensure_unique_ids(ids: list[str], what: str, path: Path) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what} ids in {path}: {duplicates}")
