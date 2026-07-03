# PLAN.md — Anchor v1

branch: `main` (fresh repo — v1 lands on main)
owner: jack
status: ready to implement

---

## 0) Summary

**One-liner:** A groundedness-evaluation harness that decomposes a RAG agent's
answer into atomic claims and verifies each against retrieved context — surfacing
hallucinated claims at the claim level — driven against a LangGraph financial-
research RAG agent as the reference system under test.

**Why now:** Hallucination in production RAG is the failure mode hiring managers
ask about directly ("reduce hallucinations in production" is a very common 2026
interview question). Claim-decomposition groundedness is the validated method
(RAGAS faithfulness, FActScore, and the 2026 FinGround paper all use it). The gap
Anchor fills: these exist as a metric inside a framework (RAGAS) or as a research
system (FinGround), not as a focused, reusable, agent-native groundedness harness
you point at your own RAG agent. Packaging + agent-native framing is the wedge.

**Who it's for:** AI engineers running RAG agents who need to know, claim by
claim, whether answers are grounded in retrieved evidence — not just a single
opaque faithfulness scalar.

**Success looks like (v1, shippable):**
- Point Anchor at a RAG agent's (answer, retrieved-context) output and get a
  claim-level groundedness report: each atomic claim marked supported /
  unsupported / contradicted, with the supporting context span, plus an overall
  groundedness score.
- A working LangGraph financial-research RAG agent as the reference system.
- Benchmark runs over a labeled set (answers with known grounded/hallucinated
  claims) showing the harness catches injected hallucinations.
- CI green: lint, format, type check, tests (all model calls mocked).

**Explicitly OUT of scope for v1 (do NOT build):**
- Grounded regeneration / auto-fixing hallucinated spans (FinGround does this;
  Anchor v1 DETECTS, it doesn't rewrite). (v2)
- World-truth fact-checking (that's factuality, a different metric — see D3).
- A dashboard/UI. Claim-level markdown/structured report only. (v2)
- Online/production-traffic scoring. Offline against a fixed set. (v2)
- A novel taxonomy — reuse the established supported/unsupported/contradicted
  verdicts; don't invent one.

FLOOR if time runs short: decompose an answer into atomic claims, verify each
against provided context with a stubbable judge, emit a claim-level report with a
groundedness score. That is complete and demoable. Everything else is additive.

---

## 1) Architecture & key decisions

Made. Do not re-litigate mid-build; if one is wrong, STOP and flag.

**D1 — Two decoupled halves: the harness (product) and the reference RAG agent
(subject).** The harness scores any (answer, context) pair from any conforming
agent; the financial-research agent is one reference implementation behind an
interface.

**D2 — Three-stage groundedness pipeline: extract → verify → score.** Claim
extraction (answer -> atomic claims), per-claim verification (claim + context ->
verdict + evidence span), scoring (verdicts -> groundedness score). Each stage is
independent and independently testable. This mirrors the validated RAGAS/FActScore
method.

**D3 — STRICT context-only grounding.** A claim is grounded iff supported by the
RETRIEVED CONTEXT. A claim true in the world but absent from context is
UNSUPPORTED, not grounded. Groundedness ≠ factuality; do not conflate them. This
is the single most important semantic decision — getting it wrong makes the metric
meaningless.

**D4 — The judge is INJECTED as a thin model client, not a high-level API.** The
judge interface is a completion client (prompt -> structured output). The prompts
and the output parsing — the actual groundedness method — live in the harness
(`extract.py`, `verify.py`), in one place. Swapping judges means swapping a
client, never reimplementing the method. (A high-level
`extract_claims`/`verify_claim` judge interface would reduce extract/verify to
pass-through wrappers and make their stub-based tests circular.) Tests stub the
client with canned raw outputs, including malformed ones.

**D5 — Claim-level output, not a scalar.** Per-claim verdict + which context span
supports it. Localization is the product's value; a single number is what already
exists elsewhere.

**D6 — Reference agent is agentic RAG on LangGraph; checkpointer INJECTED.** Not
one-shot retrieve-then-summarize: retrieve, assess sufficiency, retrieve again if
needed, answer with citations. Typed state schema with reducers. Checkpointer is
injected like the judge: SqliteSaver default (zero-setup demo path), PostgresSaver
supported and documented as the durable path, MemorySaver only inside tests.

**D7 — `src/` layout, `pyproject.toml` single source of truth, `uv` toolchain.**

**D8 — Evidence spans are quote-then-locate.** The judge returns a verbatim quote
plus a chunk reference; the harness locates the quote in the actual context and
fails loudly if it isn't there (a fabricated "supporting quote" is itself a
caught failure, not a silent pass). Never model-produced character offsets — LLMs
are unreliable at offsets.

**D9 — Harness core stays lightweight; agent deps are optional extras.**
langgraph, the vector store, and checkpointer drivers live behind an
`anchor[agent]` extra. Installing the harness alone must not drag in the
reference agent's stack — it's a harness you point at YOUR agent.

**D10 — Zero extracted claims is a loud error, not a score.** `supported/total`
over an empty claim list means extraction broke upstream; raise, don't return
0 or 1.

---

## 2) Core contracts (define FIRST, before the pipeline or agent)

**Claim schema.** An atomic claim: the claim text (one verifiable fact), a link
back to the answer span it came from, and (after verification) a verdict +
evidence reference.

**Verdict contract.** `supported | unsupported | contradicted`, plus a rationale
and the evidence (if any): chunk reference + verbatim quote, per D8 — validated
by the harness against the real context, never trusted from the judge. Fixed
vocabulary — don't expand it in v1.

**Judge interface.** A thin injected model client — essentially
`complete(prompt) -> structured output` (exact signature settled at
implementation). `extract.py` and `verify.py` own the prompts and the
parsing/validation of raw judge output; the harness depends on the client
protocol, not on a concrete model. Tests stub the client with canned raw
outputs (including malformed ones) to exercise the parsing paths.

**RAG-agent-under-test interface.** Given a question, run the agent and return
(answer, retrieved_context). One concrete implementation (the financial-research
agent) for v1.

If any of these change during the build, that's a STOP-and-flag.

---

## 3) Groundedness method (the validated pipeline — implement this)

Anchor on the RAGAS-faithfulness / FActScore approach, adapted as a harness:

1. **Claim extraction.** Decompose the agent's answer into atomic claims — each a
   single, independently verifiable statement. (An answer sentence often yields
   multiple atomic claims; split them.)
2. **Per-claim verification.** For each claim, judge it against the retrieved
   context ONLY: supported (entailed by a context span), unsupported (no span
   supports it), or contradicted (a span contradicts it). Capture the span.
3. **Scoring.** Groundedness = supported claims / total claims. Report the
   breakdown (counts per verdict) alongside the score. Unsupported + contradicted
   claims are the surfaced hallucinations. Zero claims -> loud error (D10).

Notes grounded in the research:
- Claim decomposition is computationally intensive and judge-dependent — expect
  variance across judge models; that's expected, and is why the judge is injected
  and swappable rather than fixed.
- Redundant/conflicting context can cause over-flagging of faithful answers; note
  this as a known limitation in the README rather than pretending the metric is
  perfect. Context-faithfulness auditing is an open problem, not a solved one —
  position Anchor honestly as an auditing aid.
- The context-only rule (D3) lives in the VERIFICATION PROMPT. Mocked unit tests
  cannot prove it — they prove plumbing, parsing, span validation, and scoring
  math. D3 is demonstrated by the live-judge benchmark run (§6 step 7); the
  README and test docs must not imply the unit tests guarantee it.

---

## 4) The reference financial-research agent (subject under test)

Bounded but real. The agent's job: answer a question about a company's financials
by retrieving from a small corpus of filings-style documents and producing a
cited answer.

- Agentic RAG on LangGraph: retrieve candidate chunks, assess whether they're
  sufficient, retrieve again if not, then synthesize a cited answer. Tools: a
  retriever over the document corpus; optionally a re-retrieve/refine step.
- State: accumulating (messages, retrieved chunks, claims); current (current
  question, sufficiency flag). Injected checkpointer per D6. Prune
  retrieved-chunk state.
- Corpus: a small set of SYNTHETIC or public filings-style documents. Do NOT use
  proprietary data. The corpus needs enough substance that answers contain
  multiple claims (some groundable, some temptingly not) — that's what makes the
  groundedness eval meaningful.

The agent is the thing that *produces* answers to score; Anchor is the thing that
scores their groundedness.

---

## 5) Package structure (target layout)

```
anchor/
  pyproject.toml
  README.md
  CLAUDE.md
  PLAN.md
  src/anchor/
    __init__.py            # public API only
    claim.py               # Claim + Verdict schemas
    extract.py             # answer -> list[Claim] (via injected judge)
    verify.py              # (claim, context) -> Verdict (via injected judge)
    score.py               # verdicts -> groundedness score + breakdown
    judge.py               # judge client protocol + model-backed impls
    report.py              # claim-level markdown/structured report
    agent/                 # the reference financial-research RAG agent (subject)
      __init__.py
      graph.py             # LangGraph StateGraph, agentic-RAG loop
      retriever.py         # retriever over the document corpus
      state.py             # typed state schema + reducers
    interface.py           # RAG-agent-under-test interface (keeps harness reusable)
    cli.py                 # run agent over questions -> score groundedness -> report
  tests/
    test_claim.py
    test_extract.py          # stubbed judge -> known claim list
    test_verify.py           # stubbed judge -> known verdicts (support/unsupport/contradict)
    test_score.py            # verdicts -> known groundedness score
    test_report.py
    test_agent_smoke.py      # agentic-RAG loop runs over a fixture, mocked LLM
  data/
    corpus/                  # small synthetic/public filings-style docs
    benchmark/               # labeled answers with known grounded/hallucinated claims
```

`pyproject.toml` defines the `anchor[agent]` extra (D9): harness deps in core,
agent stack (langgraph, vector store, checkpointer drivers) optional.

Filenames are the target. Justified structural changes are a STOP-and-flag.

---

## 6) Implementation order (TDD — interleave tests, never batch at end)

Write each unit's test BEFORE its implementation. Passing test = move on.

1. **Schemas.** `test_claim.py`: Claim + Verdict construct and validate; verdict
   vocabulary fixed. Implement `claim.py`.
2. **Judge client + stub.** Define `judge.py` (client protocol + model-backed
   impl). Build a deterministic stub client for tests (canned raw outputs,
   including malformed ones). No real model calls anywhere in tests.
3. **Extract → Verify → Score, each test-first.**
   - `test_extract.py`: stub client returns canned raw extraction output —
     well-formed AND malformed; implement `extract.py` (prompt construction,
     parsing, validation, answer-span linking).
   - `test_verify.py`: stub client returns canned raw verdict outputs; implement
     `verify.py` (parsing, verdict validation, quote-then-locate evidence
     validation — a quote absent from the context -> loud failure, D8). NOTE:
     these tests prove plumbing and parsing, NOT the context-only rule — D3
     lives in the verification prompt and is demonstrated only by the
     live-judge benchmark run (step 7).
   - `test_score.py`: known verdict lists → known groundedness score + breakdown;
     implement `score.py`.
4. **Report.** `test_report.py`: claim-level report renders — each claim, verdict,
   supporting span; overall score + breakdown. Implement `report.py`.
5. **Reference agent.** `test_agent_smoke.py` with a MOCKED LLM. Implement the
   agentic-RAG LangGraph agent (`state.py`, `retriever.py`, `graph.py`). Verify
   reducers + injected checkpointer + the retrieve/assess/re-retrieve loop. No
   live calls.
6. **Interface.** `interface.py` defines the RAG-agent-under-test contract with the
   research agent as the reference impl. Test against a canned run.
7. **Corpus + benchmark.** Author the small synthetic corpus in `data/corpus/` and
   a labeled benchmark in `data/benchmark/` — answers with known grounded and
   hallucinated claims, so you can show the harness catches the planted ones.
   Matching criterion (the extractor will decompose answers differently than the
   labels, so exact claim matching is impossible): a planted hallucination counts
   as CAUGHT iff at least one extracted claim overlapping its labeled answer span
   gets unsupported or contradicted. This live-judge run is also what
   demonstrates D3.
8. **CLI + README.** Thin `cli.py`: run agent over questions, score groundedness,
   emit report. README leads with the wedge: a reusable, agent-native groundedness
   harness (claim-level, not a scalar), built on the validated RAGAS/FActScore
   method, honestly positioned as an auditing aid; not a FinGround reimplementation.
   Show a sample report catching a hallucinated claim. Cite RAGAS/FActScore/FinGround
   as the methodological basis.

---

## 7) Verification before calling it done

Show evidence, not assertions:
- `uv run pytest` all green — confirm NO real network/model calls in tests.
- `uv run ruff check .` and `mypy src/` clean.
- `uv build` succeeds; in a fresh venv, install the wheel and run the CLI over the
  benchmark producing a claim-level report that catches the planted hallucinations
  (per the §6 step-7 matching criterion). Also confirm a bare `anchor` install
  does not pull in the agent stack (D9).
- Confirm scope: detection only (no regeneration), context-only grounding (not
  factuality), judge injected (not hardcoded), agent not welded to the harness.

Fix and re-verify on any failure. Never report done on unverified work.

---

## 8) Notes for the implementing model

- This spec is the source of truth. Diverge from reality → STOP and say so.
- Load-bearing seams: **claim schema**, **verdict contract**, **judge interface**,
  **RAG-agent interface**, **LangGraph state schema**. Guard them.
- The context-only grounding rule (D3) is the semantic linchpin. Groundedness is
  traceability to retrieved evidence, NOT world-truth. Test it explicitly.
- The judge is injected as a thin model client; the prompts and parsing are the
  harness's core and live in extract/verify. Never hardcode a model.
- Anchor DETECTS in v1; it does not regenerate. Detection is the whole v1.
- Be honest in the README: claim-decomposition groundedness is judge-dependent and
  imperfect (over-flags on redundant/conflicting context). Position as an auditing
  aid, citing the method's origins — that honesty reads as maturity, not weakness.
- Reference agent + corpus stay synthetic/public and bounded. Anchor is an eval
  harness, not a real financial-research product. No proprietary data.
- Boring, explicit, loud failures. Mock everything external in tests.
