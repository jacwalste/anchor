# Anchor — Design Walkthrough

Companion to [PLAN.md](PLAN.md) (the build spec). This document explains the
system as built: what each piece is, the decisions behind it, and the
tradeoffs those decisions bought. If you can answer the questions at the
bottom without looking, you understand the project.

## The 30-second version

RAG agents hallucinate relative to their own evidence: they state things the
retrieved documents don't back up. Anchor measures that. It decomposes an
answer into atomic claims, verifies each claim against the retrieved context
only, and reports per-claim verdicts (supported / unsupported / contradicted)
with mechanically-validated evidence quotes. The score is supported ÷ total;
the per-claim localization is the product. A LangGraph financial-research
agent ships as the reference system under test, behind an interface, so the
harness points at any RAG agent.

## System map

```
             HARNESS (the product)                    SUBJECT (the reference agent)
┌──────────────────────────────────────────┐   ┌─────────────────────────────────────┐
│ evaluate_answer()          evaluate.py   │   │ FinancialResearchAgent   adapter.py │
│   ├─ extract.py   answer -> [Claim]      │   │   └─ graph.py   LangGraph loop:     │
│   ├─ verify.py    claim+ctx -> Verdict   │   │        retrieve -> assess ──┐       │
│   ├─ score.py     verdicts -> score      │   │            ^── refined query┘       │
│   └─ report.py    -> markdown            │   │        -> answer (cited)            │
│                                          │   │   state.py      reducers            │
│ claim.py     the shared vocabulary       │   │   retriever.py  lexical, injected   │
│ judge.py     JudgeClient (prompt->text)  │   │                                     │
│ benchmark.py planted-hallucination runner│   │                                     │
└──────────────────────────────────────────┘   └─────────────────────────────────────┘
                     interface.py — RAGAgent protocol + AgentRun
              (the ONLY seam; adapter.py is the only file touching both)
```

Judge implementations (`judge_anthropic.py`) and the agent stack are optional
extras; the core installs with pydantic alone.

## The pipeline, stage by stage

**Extract** (`extract.py`) — one judge call. Owns the extraction prompt
(what "atomic" means) and all parsing. Three gates before anything becomes a
`Claim`: valid JSON → valid schema → `source_text` appears verbatim in the
answer. That last gate is provenance: the judge *asserts* where a claim came
from; the harness verifies it with `in`, not with trust.

**Verify** (`verify.py`) — one judge call per claim. Owns the verification
prompt, which is where the context-only rule (D3) physically lives. Same
three gates, plus two more on evidence: the cited chunk id must resolve to a
chunk we actually provided, and the quote must appear verbatim inside it
(quote-then-locate). A judge inventing a plausible-sounding quote is caught
by string search, not by another model.

**Score** (`score.py`) — no judge. Stores only the three verdict counts;
`total`, `score`, and `hallucinated` are computed properties, so the numbers
cannot disagree. Zero claims raises: you can *find* nothing (extraction may
legitimately return `[]` for "I don't know"), but you cannot *score* nothing.

**Report** (`report.py`) — no judge. Flagged claims render first because the
user's question is "which sentence do I distrust?" — the scalar is a summary
line, not the product. Takes only the verified claims and computes the score
itself, so a report can never display a score that disagrees with its own
claim list.

## Decision record

Each entry: the decision, why, and what it cost.

**1. Groundedness ≠ factuality (D3).** A claim true in the world but absent
from the retrieved context is *unsupported*. The metric measures traceability
to evidence, not truth. Cost: the metric looks "unfair" to knowledgeable
answers — that's the point, and the benchmark's headquarters case exists to
demonstrate it. This is the single most load-bearing semantic decision;
getting it wrong makes the number meaningless.

**2. The judge is a thin completion client, not a high-level API.**
`JudgeClient` is `complete(prompt) -> str`. The prompts and parsing — the
actual method — live in `extract.py`/`verify.py`, once. The rejected
alternative (`judge.extract_claims()` / `judge.verify_claim()`) would have
made the pipeline stages pass-through wrappers, their stub-based tests
circular (asserting a stub returns what the stub returns), and every new
judge model a reimplementation of the method. Implementations may still use
provider structured-output features internally; the harness validates every
response regardless.

**3. Quote-then-locate, on both sides.** Models produce quotes; code computes
positions. Claims carry a verbatim `source_text` (located in the answer);
evidence carries a verbatim quote + chunk id (located in the chunk). Never
model-produced character offsets — LLMs are unreliable at offsets, and a
verbatim string is mechanically checkable.

**4. Invalid states are unrepresentable.** Three applications: a `Verdict`
refuses to construct as supported/contradicted without evidence or as
unsupported *with* evidence (an unsupported verdict citing evidence is
self-contradictory); `VerifiedClaim` is a separate type rather than a
nullable `verdict` field on `Claim`, so the scorer's signature makes
"has this been verified?" unaskable; derived numbers are computed properties,
never stored fields. The common thread: encode the invariant in the type so
no downstream code has to check it.

**5. Two exception types, sorted by fault.** `ValueError` = the caller's bug
(empty answer, empty context, bad concurrency) — crash fast, never catch.
`JudgeResponseError` = the model dependency misbehaving at runtime — an
expected operational failure that carries the raw response and that batch
callers legitimately catch per-item (the benchmark runner does exactly this:
one failed case is recorded, the run continues).

**6. Zero silent retries.** Malformed judge output raises immediately with
the raw response attached. Retry machinery gets added when live runs produce
evidence of a real malformed-output rate, not before. The only parsing
leniency anywhere is stripping a markdown code fence — a deterministic
transform, unlike fishing JSON out of prose, which is interpretive guessing
at a trust boundary.

**7. Threads, not async; sequential by default.** Per-claim verification is
independent, I/O-bound HTTP against a sync SDK — a `ThreadPoolExecutor` in
`evaluate_answer` parallelizes it without infecting every signature with
`async`. The *library* defaults to sequential because concurrency is only
safe with a thread-safe judge and the library can't know what was injected;
the *CLI* defaults to 4 because it always constructs the thread-safe
`AnthropicJudge`.

**8. Optional extras with import discipline.** Core = pydantic only.
langgraph lives behind `anchor[agent]`, the Anthropic SDK behind
`anchor[anthropic]`, and core never imports either module eagerly — so bare
`import anchor` works without them and `import anchor.agent` fails with an
actionable install hint. Verified in a fresh venv as part of acceptance.

**9. The reference agent is agentic RAG with reducers.** Not one-shot
retrieve-then-summarize: retrieve → model assesses sufficiency → optionally
re-retrieve with a refined query (hard-capped rounds) → answer with chunk-id
citations. LangGraph state fields that accumulate (`retrieved`, `queries`,
`rounds`) carry explicit reducers — including a dedup-by-id merge for chunks —
because a missing reducer means later rounds silently overwrite earlier
state, the classic LangGraph failure. Current-state fields (`sufficient`,
`answer`) legitimately overwrite. Checkpointer, model, and retriever are all
injected; the v1 retriever is deliberately lexical (IDF-weighted overlap,
zero dependencies) behind a `Retriever` protocol.

**10. The benchmark plants hallucinations and defines "caught."** Labeled
answers embed known-bad spans (fabricated facts, a contradicted trend, a
world-plausible-but-absent fact). Because the extractor decomposes answers
differently than the labels, exact matching is impossible; a planted span
counts as caught iff at least one extracted claim whose `source_text`
overlaps it (by character range) received a non-supported verdict. A
non-overlapping flagged claim does not count — no credit for coincidence.
The data files are validated by their loaders *in the test suite*, so an
authoring typo fails CI.

**11. Mocked tests prove plumbing, not the metric.** The suite (91 tests, no
network) proves parsing, validation gates, scoring math, graph mechanics, and
CLI wiring. It cannot prove the context-only rule — that lives in a prompt,
and only a live-judge benchmark run demonstrates it. The README says this
plainly; pretending unit tests validate an LLM-judged metric would be the
exact dishonesty the tool exists to catch.

## Deliberately not built

Grounded regeneration (detect-only is v1's whole identity), any UI, online
production scoring, a vector-store retriever (the protocol proves
swappability; the dependency would demonstrate nothing), and a novel verdict
taxonomy (supported/unsupported/contradicted is established vocabulary —
inventing a fourth category would be differentiation theater). A focused tool
that says no clearly is the deliverable.

## Known limitations

Judge-dependence is the big one: different judges extract and verify
differently, so results should always name the judge (and the roadmap answer
is a judge-variance table, not a pretense of judge-independence). Redundant
or conflicting context can over-flag faithful answers. Overlap matching uses
first occurrence of repeated spans. The lexical retriever is deliberately
simple — retrieval quality is not what this project demonstrates.

## Check yourself

If this project is yours, you should be able to answer these cold:

1. Why is a world-true claim marked unsupported, and why is that a feature?
2. Why would `judge.extract_claims()` have made `test_extract.py` worthless?
3. What are the five gates between a judge's verify response and a
   `VerifiedClaim`, and which failure each one catches?
4. Why is `VerifiedClaim` a separate type instead of `Claim.verdict: Verdict | None`?
5. When may a caller catch `JudgeResponseError`, and why must they never
   catch `ValueError` from this library?
6. Why do reducers exist in the agent state, and what silently breaks
   without the dedup merge?
7. Why does the benchmark need an *overlap* criterion instead of comparing
   claims to labels directly?
8. What do the 91 mocked tests prove, and what can they never prove?
