# Anchor

Claim-level groundedness evaluation for RAG agents.

Anchor takes a RAG agent's answer and the context it retrieved, decomposes the
answer into atomic claims, and verifies each claim against that context —
surfacing exactly which claims are supported, unsupported (hallucinated), or
contradicted, with the evidence for each verdict.

**Status: early development.** See [PLAN.md](PLAN.md) for the full spec.

## Why

RAG systems hallucinate: they state things their retrieved evidence doesn't
support. Existing tools give you a single faithfulness scalar; Anchor gives you
a claim-level report — *which* statement is ungrounded and *where* the evidence
gap is. Groundedness here means traceability to retrieved context, not
world-truth: a claim that is true but absent from the context is still
unsupported.

The method (claim decomposition + per-claim verification) follows RAGAS
faithfulness and FActScore. Anchor packages it as a reusable harness you point
at your own agent, with an injected judge model and honest limitations:
claim-level judgments are judge-dependent — treat this as an auditing aid.

## Development

```sh
uv sync           # install
uv run pytest     # test (no live model calls, ever)
uv run ruff check .
uv run mypy src/
```
