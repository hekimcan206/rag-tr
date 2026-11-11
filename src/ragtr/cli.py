"""Command line: generate the synthetic data set, migrate, ingest, ask, and run evaluation suites."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from ragtr.data import synth

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Turkish RAG toolkit.")


@app.command("generate-data")
def generate_data(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("data"),
    seed: Annotated[int, typer.Option()] = synth.SEED,
) -> None:
    """Write the synthetic corpus (Markdown) and QA set (JSONL). Same seed, same bytes."""
    docs, qas = synth.write(out, seed)
    typer.echo(f"wrote {docs} documents and {qas} questions to {out}")


@app.command()
def migrate(
    app_role: Annotated[str | None, typer.Option(help="Role that receives DML grants")] = None,
) -> None:
    """Create the schema as the owner role (RAGTR_ADMIN_DATABASE_URL) and grant DML to the app role."""
    from ragtr.config import Settings, build_components  # noqa: PLC0415
    from ragtr.store.postgres import migrate as run_migrate  # noqa: PLC0415

    settings = Settings()
    comps = build_components(settings)
    url = os.environ.get("RAGTR_ADMIN_DATABASE_URL") or settings.database_url.get_secret_value()
    run_migrate(url, comps.embedder.dim, comps.embedder.name, comps.analyzer.name, app_role)
    typer.echo(f"schema ready (embedder={comps.embedder.name}, dim={comps.embedder.dim}, "
               f"analyzer={comps.analyzer.name})")  # fmt: skip


@app.command()
def ingest(
    data: Annotated[Path, typer.Argument(help="Data directory containing corpus/*.md")] = Path("data"),
    api: Annotated[str, typer.Option(help="Base URL of the rag-tr API")] = "http://localhost:8000",
) -> None:
    """Upload every Markdown file through the HTTP API (PUT is idempotent: re-running replaces)."""
    from ragtr.data.dataset import load_corpus  # noqa: PLC0415

    docs = load_corpus(data)
    total = 0
    with httpx.Client(base_url=api, timeout=300) as client:
        for doc in docs:
            r = client.put(f"/v1/documents/{doc.id}", json={"title": doc.title, "text": doc.text,
                                                            "source": doc.source})  # fmt: skip
            r.raise_for_status()
            total += int(r.json()["chunks"])
    typer.echo(f"ingested {len(docs)} documents, {total} chunks")


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    api: Annotated[str, typer.Option()] = "http://localhost:8000",
) -> None:
    """Ask a question and print the answer with its sources."""
    with httpx.Client(base_url=api, timeout=300) as client:
        r = client.post("/v1/ask", json={"question": question})
        r.raise_for_status()
        body = r.json()
    typer.echo(body["answer"])
    for c in body["citations"]:
        typer.echo(f"  [{c['marker']}] {c['title']} > {c['section']}  ({c['chunk_id']})")


@app.command("eval")
def evaluate(
    suite: Annotated[str, typer.Option(help="main | chunking | normalisation | embedders | generation | ci")],
    data: Annotated[Path, typer.Option()] = Path("data"),
    embedder: Annotated[str, typer.Option()] = "e5-small",
    embedders: Annotated[str, typer.Option(help="Comma-separated, for the embedders suite")] = "",
    reranker: Annotated[str, typer.Option()] = "none",
    generator: Annotated[str, typer.Option(help="extractive | ollama | none")] = "extractive",
    ollama_url: Annotated[str, typer.Option()] = "http://localhost:11434",
    ollama_model: Annotated[str, typer.Option()] = "gemma3:4b",
    limit: Annotated[int | None, typer.Option(help="Stratified sample size for answer metrics")] = None,
    out: Annotated[Path, typer.Option()] = Path("eval/results"),
    compare: Annotated[Path | None, typer.Option(help="Fail if metrics differ from this result file")] = None,
) -> None:
    """Run an evaluation suite on the committed data set and write JSON + Markdown results."""
    from ragtr.data.dataset import load_corpus, load_qa  # noqa: PLC0415
    from ragtr.eval.harness import run_suite  # noqa: PLC0415
    from ragtr.eval.report import render  # noqa: PLC0415
    from ragtr.generation import make_generator  # noqa: PLC0415
    from ragtr.rerank import make_reranker  # noqa: PLC0415

    gen = None if generator == "none" else make_generator(generator, ollama_url, ollama_model, 600.0)
    result = run_suite(
        suite,
        load_corpus(data),
        load_qa(data),
        embedder=embedder,
        reranker=make_reranker(reranker),
        generator=gen,
        embedders=[e for e in embedders.split(",") if e],
        answer_limit=limit,
        progress=lambda msg: typer.echo(msg, err=True),
    ).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{suite}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{suite}.md").write_text(render(result), encoding="utf-8")
    typer.echo(render(result))
    if compare is not None:
        diffs = _metric_diffs(json.loads(compare.read_text(encoding="utf-8")), result)
        if diffs:
            typer.echo("metrics differ from " + str(compare) + ":\n  " + "\n  ".join(diffs), err=True)
            sys.exit(1)
        typer.echo(f"metrics identical to {compare}")


def _metric_diffs(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Compare everything except timings and environment, which legitimately vary between machines."""

    def strip(run: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in run.items() if k != "seconds"}
        if out.get("answers"):
            out["answers"] = {k: v for k, v in out["answers"].items() if not k.startswith("latency")}
        return out

    exp = {r["name"]: strip(r) for r in expected["runs"]}
    act = {r["name"]: strip(r) for r in actual["runs"]}
    diffs = [f"run {n!r} missing" for n in exp.keys() - act.keys()]
    diffs += [f"unexpected run {n!r}" for n in act.keys() - exp.keys()]
    diffs += [f"{n}: expected {exp[n]} got {act[n]}" for n in exp.keys() & act.keys() if exp[n] != act[n]]
    return sorted(diffs)


if __name__ == "__main__":
    app()
