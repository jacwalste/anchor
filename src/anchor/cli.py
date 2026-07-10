"""Command line: run the labeled benchmark, or ask the reference agent and score it.

Both commands make LIVE model calls (agent and/or judge) — the only place in
the project that does. Extras load lazily so `anchor --help` works without them.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from anchor.benchmark import run_benchmark
from anchor.dataset import load_benchmark, load_chunks
from anchor.evaluate import evaluate_answer
from anchor.judge import JudgeClient
from anchor.report import render_benchmark_markdown, render_markdown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anchor", description="Claim-level groundedness evaluation for RAG agents."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    bench = subcommands.add_parser(
        "benchmark", help="run the labeled benchmark against a live judge"
    )
    bench.add_argument("--corpus", type=Path, required=True, help="corpus JSON file")
    bench.add_argument("--cases", type=Path, required=True, help="benchmark cases JSON file")
    bench.add_argument("--judge-model", required=True, help="Anthropic model id for the judge")
    bench.add_argument(
        "--max-concurrency", type=int, default=4, help="parallel verification calls per answer"
    )

    ask = subcommands.add_parser(
        "ask", help="ask the reference agent a question and score its answer"
    )
    ask.add_argument("--corpus", type=Path, required=True, help="corpus JSON file")
    ask.add_argument("--question", required=True)
    ask.add_argument("--agent-model", required=True, help="Anthropic model id for the agent")
    ask.add_argument("--judge-model", required=True, help="Anthropic model id for the judge")
    ask.add_argument(
        "--max-concurrency", type=int, default=4, help="parallel verification calls per answer"
    )

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    return _cmd_ask(args)


def _make_judge(model: str) -> JudgeClient:
    from anchor.judge_anthropic import AnthropicJudge

    return AnthropicJudge(model=model)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    result = run_benchmark(
        cases=load_benchmark(args.cases),
        corpus=load_chunks(args.corpus),
        judge=_make_judge(args.judge_model),
        max_concurrency=args.max_concurrency,
    )
    print(render_benchmark_markdown(result))
    return 1 if result.errored_case_ids else 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from anchor.agent import FinancialResearchAgent

    corpus = load_chunks(args.corpus)
    agent = FinancialResearchAgent(corpus, model=_make_judge(args.agent_model).complete)
    run = agent.run(args.question)
    print(f"## Agent answer\n\n{run.answer}\n")
    verified = evaluate_answer(
        run.answer,
        run.retrieved,
        _make_judge(args.judge_model),
        max_concurrency=args.max_concurrency,
    )
    print(render_markdown(verified))
    return 0
