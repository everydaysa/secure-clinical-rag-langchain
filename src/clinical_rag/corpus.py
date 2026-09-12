"""Load the synthetic policy corpus into LangChain documents.

Each markdown file carries YAML frontmatter describing what it is and who may
read it. That frontmatter becomes chunk metadata, which is what later steps
filter and cite on.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .access import access_metadata

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

REQUIRED_FIELDS = ("doc_id", "title", "doc_type", "version", "allowed_roles")

DEFAULT_CORPUS_DIR = Path("data/corpus")


class CorpusError(ValueError):
    """Raised when a corpus file is missing required metadata."""


def parse_file(path: Path) -> tuple[dict, str]:
    """Split one markdown file into (frontmatter dict, body text)."""
    match = FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise CorpusError(f"{path.name}: missing YAML frontmatter block")

    meta = yaml.safe_load(match.group(1)) or {}
    missing = [field for field in REQUIRED_FIELDS if field not in meta]
    if missing:
        raise CorpusError(f"{path.name}: missing frontmatter fields: {missing}")
    return meta, match.group(2)


def _section_of(chunk: str, fallback: str) -> str:
    """Best-effort section label for a chunk, used in citations."""
    for line in chunk.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return fallback


def load_corpus(
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Return chunked documents with access and citation metadata attached."""
    corpus_dir = Path(corpus_dir)
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise CorpusError(f"No markdown files found in {corpus_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    documents: list[Document] = []
    for path in paths:
        meta, body = parse_file(path)
        base = {
            "doc_id": str(meta["doc_id"]),
            "title": str(meta["title"]),
            "doc_type": str(meta["doc_type"]),
            "version": str(meta["version"]),
            "effective_date": str(meta.get("effective_date", "")),
            "owner": str(meta.get("owner", "")),
            "sensitivity": str(meta.get("sensitivity", "internal")),
            "allowed_roles": ",".join(meta["allowed_roles"]),
            "source": path.name,
            **access_metadata(list(meta["allowed_roles"])),
        }

        for index, chunk in enumerate(splitter.split_text(body)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **base,
                        "chunk_index": index,
                        "section": _section_of(chunk, meta["title"]),
                    },
                )
            )

    return documents
