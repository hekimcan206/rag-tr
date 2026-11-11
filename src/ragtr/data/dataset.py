"""Load the committed corpus (``data/corpus/*.md``) and QA set (``data/qa.jsonl``)."""

from __future__ import annotations

import json
from pathlib import Path

from ragtr.data.synth import QA
from ragtr.models import Document


def load_corpus(data_dir: Path) -> list[Document]:
    """One document per Markdown file; the id is the file name, the title its first ``#`` heading."""
    documents: list[Document] = []
    for path in sorted((data_dir / "corpus").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first = text.lstrip().splitlines()[0] if text.strip() else ""
        title = first.lstrip("#").strip() if first.startswith("#") else path.stem
        documents.append(Document(path.stem, title, text, source=f"corpus/{path.name}"))
    return documents


def load_qa(data_dir: Path) -> list[QA]:
    items: list[QA] = []
    with (data_dir / "qa.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(QA(**json.loads(line)))
    return items
