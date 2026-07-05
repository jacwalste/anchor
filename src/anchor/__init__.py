"""Anchor: claim-level groundedness evaluation for RAG agents."""

from anchor.claim import Chunk, Claim, Evidence, Verdict, VerdictLabel, VerifiedClaim
from anchor.dataset import (
    BenchmarkCase,
    PlantedHallucination,
    load_benchmark,
    load_chunks,
    resolve_context,
)
from anchor.errors import JudgeResponseError
from anchor.evaluate import evaluate_answer
from anchor.extract import extract_claims
from anchor.interface import AgentRun, RAGAgent
from anchor.judge import JudgeClient
from anchor.report import render_markdown
from anchor.score import GroundednessScore, score_claims
from anchor.verify import verify_claim

__all__ = [
    "AgentRun",
    "BenchmarkCase",
    "Chunk",
    "Claim",
    "Evidence",
    "GroundednessScore",
    "JudgeClient",
    "JudgeResponseError",
    "PlantedHallucination",
    "RAGAgent",
    "Verdict",
    "VerdictLabel",
    "VerifiedClaim",
    "evaluate_answer",
    "extract_claims",
    "load_benchmark",
    "load_chunks",
    "render_markdown",
    "resolve_context",
    "score_claims",
    "verify_claim",
]
