# CLAUDE.md — Anchor

Groundedness-evaluation harness for RAG agents. Decomposes an agent's answer into
atomic claims and verifies each against retrieved source context, surfacing
unsupported (hallucinated) claims. Driven against a LangGraph financial-research
RAG agent as the reference system under test.

This file is a behavioral contract, not documentation. Every line should change
how you act. Non-negotiable rules live in CI/pre-commit, not here.

## What this project is (and is not)
- IS: an eval library you point at a RAG agent to score groundedness — extract
  atomic claims from the answer, classify each as supported/unsupported/contradicted
  against retrieved context, report claim-level verdicts + a groundedness score.
- IS ALSO: a real LangGraph financial-research RAG agent as the reference system
  the harness evaluates.
- IS NOT: a reimplementation of a research paper. The claim-decomposition method
  is established (RAGAS faithfulness, FActScore, FinGround). Anchor's contribution
  is packaging it as a REUSABLE HARNESS you point at any RAG agent — FinGround is a
  system, RAGAS's faithfulness is one metric inside a framework; Anchor is a
  focused, agent-native groundedness evaluator. If a change turns this into a
  paper clone, flag it.

## Commands
- Install (dev): `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy src/`
- Build: `uv build`

## Architecture (read `PLAN.md` for the full spec before implementing)
- `src/anchor/`, `src/` layout. Two decoupled halves:
  (1) the GROUNDEDNESS HARNESS (product), (2) the reference RAG AGENT (subject).
- Harness pieces: a claim extractor (answer -> list of atomic claims), a verifier
  (claim + retrieved context -> supported/unsupported/contradicted + rationale),
  a scorer (verdicts -> groundedness score), and a reporter. Each independently
  testable.
- The judge is INJECTED as a thin model client (prompt -> structured output),
  never hardcoded. Prompts + output parsing — the actual method — live in the
  harness (extract/verify), not in judge implementations.
- The RAG-agent-under-test is accessed through an INTERFACE. Ship the financial-
  research agent as the reference implementation behind it.
- Reference agent uses LangGraph. State schema is the most consequential decision.

## Conventions
- `uv` for env/packages; `pyproject.toml` single source of truth.
- `ruff` lint + format. `pytest`. `mypy` on `src/`. Conventional Commits.
- Boring and explicit over clever. Fail loudly, never silently.
- Tests: extractor/verifier/scorer get unit tests with a STUBBED judge client
  (canned raw outputs, well-formed and malformed). Mock all real model calls —
  no live calls in tests, ever. Mocked tests prove plumbing/parsing/scoring,
  not the context-only rule — that's demonstrated by the live-judge benchmark.

## LangGraph non-negotiables (reference agent)
- Typed state schema with explicit reducers. `Annotated[list, add]` for
  accumulating fields (messages, retrieved chunks, claims); plain types for
  current state. Silent overwrites are the #1 documented failure.
- Checkpointer: INJECTED like the judge. SqliteSaver default (zero-setup demo);
  PostgresSaver supported and documented as the durable path. MemorySaver only
  inside tests.
- Agentic RAG: the agent retrieves, checks sufficiency, retrieves again if needed,
  then answers with citations — not one-shot retrieve-then-summarize. Prune state.

## Groundedness framing (do not drift)
- Method: decompose the answer into ATOMIC claims (one verifiable fact each),
  then verify EACH claim against the retrieved context only — supported /
  unsupported / contradicted. Groundedness score = supported claims / total claims.
- STRICT context-only rule: a claim true in the world but absent from retrieved
  context is UNSUPPORTED. Groundedness measures traceability to the provided
  evidence, not world-truth. (World-truth is a different metric — don't conflate.)
- Output is claim-LEVEL (per-claim verdict + which context span supports it), not
  a single scalar. Localization is the point.

## When unsure
- If a change touches the claim schema, the verifier's verdict contract, the judge
  interface, the RAG-agent interface, or the state schema, STOP and surface it.
- If you're about to hardcode a judge model, don't — inject it.
- If you're drifting toward "reimplement FinGround," stop; the harness framing is
  the differentiation.
- State assumptions explicitly. Present alternatives; don't pick silently.

## Guardrails (also enforced in CI/pre-commit)
- No secrets/API keys/credentials committed, ever. Model creds via env.
- Tests make NO real network or model calls — stub/mock everything.
- Changes to claim schema, verdict contract, judge interface, or RAG interface
  require a test and a changelog note.
