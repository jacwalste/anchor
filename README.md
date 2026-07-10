# Anchor

**Claim-level groundedness evaluation for RAG agents.** Anchor decomposes an
agent's answer into atomic claims and verifies each one against the context
the agent actually retrieved — surfacing *which sentence* is hallucinated and
*where* the evidence gap is, not just a faithfulness scalar.

```markdown
# Groundedness Report

**Score: 0.67** — 2 of 3 claims supported

## Flagged claims

- **unsupported** — Acme leads the cloud market
  - from answer: "it leads the cloud market"
  - rationale: No chunk mentions market position.

## All claims

1. **supported** — Acme's revenue grew 12% in FY2025
   - from answer: "revenue grew 12%"
   - evidence [10k-item7-001]: "up 12%"
   ...
```

## What "grounded" means here

A claim is grounded **iff the retrieved context supports it**. A claim that is
true in the real world but absent from the context is *unsupported* —
groundedness measures traceability to evidence, not world-truth (that's
factuality, a different metric). Verdicts are `supported` / `unsupported` /
`contradicted`; the score is supported ÷ total; the per-claim breakdown is the
actual product.

Every judge citation is verified mechanically: the judge must return a
verbatim quote plus a chunk id, and Anchor locates that quote in the real
chunk — a fabricated "supporting quote" is caught as an error, not trusted.

## Design

- **The judge is injected**, as a minimal `prompt -> text` client
  (`JudgeClient`). Prompts and response parsing — the method itself — live in
  the harness, so swapping judge models never means reimplementing it. An
  Anthropic-backed judge ships behind the `anthropic` extra.
- **The agent under test sits behind an interface** (`RAGAgent`:
  `question -> answer + retrieved chunks`). Anchor ships a reference
  agent — an agentic-RAG financial-research agent on LangGraph (retrieve →
  assess sufficiency → re-retrieve if needed → answer with citations) — behind
  the `agent` extra, but the harness works against anything implementing the
  protocol.
- **Failures are loud.** Malformed judge output, fabricated quotes, citations
  of unknown chunks, empty retrievals, zero extracted claims: all raise with
  the raw response attached. No silent retries, no defaults.

## Quickstart

```sh
uv sync --all-extras
export ANTHROPIC_API_KEY=...

# Ask the reference agent a question and score its answer:
uv run anchor ask \
  --corpus data/corpus/acmecloud_fy2025.json \
  --question "How did AcmeCloud's operating margin change in fiscal 2025?" \
  --agent-model claude-sonnet-5 --judge-model claude-sonnet-5

# Run the labeled benchmark (planted hallucinations, catch-rate report):
uv run anchor benchmark \
  --corpus data/corpus/acmecloud_fy2025.json \
  --cases data/benchmark/cases.json \
  --judge-model claude-sonnet-5
```

As a library, the front door is three calls:

```python
from anchor import evaluate_answer, render_markdown, score_claims

# extract + verify; max_concurrency parallelizes the per-claim verification calls
verified = evaluate_answer(answer, retrieved_chunks, judge, max_concurrency=8)
print(score_claims(verified).score)
print(render_markdown(verified))
```

## The benchmark

`data/` contains a synthetic filings-style corpus (fictional "AcmeCloud",
fiscal 2025) and five labeled answers with **planted hallucinations**: a
fabricated market-position claim, a margin trend stated backwards from the
filing (contradiction), a dividend the filing explicitly denies, and a
world-plausible headquarters claim that simply isn't in the context — the
canonical groundedness-vs-factuality case. One fully grounded control case
checks for over-flagging.

Because the extractor decomposes answers differently than the labels, a
planted hallucination counts as **caught** iff at least one extracted claim
overlapping its answer span receives a non-supported verdict.

## Honest limitations

Claim-decomposition groundedness is a validated method (RAGAS faithfulness,
FActScore) but it is judge-dependent: different judge models extract and
verify differently, which is why the judge is injected and reported results
should name the judge used. Redundant or conflicting context can over-flag
faithful answers. The unit test suite proves the pipeline's plumbing, parsing,
and scoring — the context-only rule itself lives in the verification prompt
and is only demonstrable with a live judge on the benchmark. Treat Anchor as
an auditing aid that localizes suspicion, not an oracle.

## Method

Claim decomposition + per-claim verification follows
[RAGAS faithfulness](https://arxiv.org/abs/2309.15217) and
[FActScore](https://arxiv.org/abs/2305.14251); FinGround (2026) applies the
same family of ideas to financial RAG with grounded regeneration. Anchor's
contribution is packaging: a small, agent-native harness with strict evidence
provenance you can point at your own system.

## Development

```sh
uv sync --all-extras
uv run pytest          # no live model calls, ever
uv run ruff check .
uv run mypy src/
```
