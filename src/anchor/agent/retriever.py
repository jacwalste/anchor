"""Deterministic lexical retriever over an in-memory corpus.

Deliberately embedding-free: retrieval quality is not what Anchor
demonstrates, and a dependency-free retriever keeps the demo runnable with
no services. Anything matching the Retriever protocol can replace it.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from anchor.claim import Chunk


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[Chunk]: ...


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class LexicalRetriever:
    """Ranks chunks by IDF-weighted token overlap with the query.

    Rare terms count more than common ones; chunks sharing no tokens with
    the query are never returned. Ties break on chunk id so results are
    fully deterministic.
    """

    def __init__(self, corpus: Sequence[Chunk], k: int = 4) -> None:
        if not corpus:
            raise ValueError("cannot retrieve from an empty corpus")
        if k < 1:
            raise ValueError("k must be >= 1")
        self._chunks = [(chunk, _tokenize(chunk.text)) for chunk in corpus]
        self._k = k
        document_frequency = Counter(token for _, tokens in self._chunks for token in tokens)
        total = len(self._chunks)
        self._idf = {
            token: math.log((total + 1) / (count + 1)) + 1.0
            for token, count in document_frequency.items()
        }

    def retrieve(self, query: str) -> list[Chunk]:
        query_tokens = _tokenize(query)
        scored: list[tuple[float, Chunk]] = []
        for chunk, tokens in self._chunks:
            score = sum(self._idf[token] for token in query_tokens & tokens)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [chunk for _, chunk in scored[: self._k]]
