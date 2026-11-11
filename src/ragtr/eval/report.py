"""Render suite results as Markdown tables (pasted into README and docs/evaluation.md)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _f(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def retrieval_table(runs: Sequence[dict[str, Any]], variants: bool = True) -> str:
    head = ["run", "chunks", "words/chunk", "R@1", "R@5", "R@10", "MRR@10", "nDCG@10", "R@300w"]
    if variants:
        head += ["R@5 canonical", "R@5 paraphrase", "R@5 ascii"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in runs:
        o = run["overall"]
        words = f"{run.get('chunk_words', 0):.1f}"
        row = [run["name"], run.get("chunks"), words, o["recall_at_1"], o["recall_at_5"], o["recall_at_10"],
               o["mrr_at_10"], o["ndcg_at_10"], o.get("recall_at_budget")]  # fmt: skip
        if variants:
            row += [
                run["by_variant"].get(v, {}).get("recall_at_5") for v in ("canonical", "paraphrase", "ascii")
            ]
        lines.append("| " + " | ".join(_f(c) for c in row) + " |")
    return "\n".join(lines)


def answer_table(runs: Sequence[dict[str, Any]]) -> str:
    head = ["contexts from", "generator", "n", "accuracy", "citation precision", "supported", "groundedness",
            "false abstention", "abstention (unanswerable)", "median s"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in runs:
        a = run.get("answers")
        if not a:
            continue
        row = [run["name"], a["generator"], a["answerable"] + a["unanswerable"], a["accuracy"],
               a["citation_precision"], a["supported"], a["groundedness"], a["false_abstention"],
               a["abstention_on_unanswerable"], a["latency_s_median"]]  # fmt: skip
        lines.append("| " + " | ".join(_f(c) for c in row) + " |")
    return "\n".join(lines)


def render(result: dict[str, Any]) -> str:
    env = result["environment"]
    parts = [
        f"## Suite `{result['suite']}`",
        "",
        result["description"],
        "",
        retrieval_table(result["runs"]),
    ]
    if any(r.get("answers") for r in result["runs"]):
        parts += ["", answer_table(result["runs"])]
    parts += [
        "",
        f"Run {env.get('timestamp_utc')} · Python {env.get('python')} · "
        f"{env.get('cpu_count')} CPUs visible · "
        f"load avg start {env.get('loadavg_start')} end {env.get('loadavg_end')} · "
        f"wall {env.get('wall_seconds')} s · embedder `{env.get('embedder')}` · "
        f"reranker `{env.get('reranker')}` · generator `{env.get('generator')}`",
    ]
    return "\n".join(parts) + "\n"
