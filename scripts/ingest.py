"""Build the search index from the synthetic policy corpus.

Run:  python scripts/ingest.py
      LLM_PROVIDER=mock python scripts/ingest.py   # offline, no Azure calls
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.config import get_settings  # noqa: E402
from clinical_rag.corpus import load_corpus  # noqa: E402
from clinical_rag.vectorstore import (  # noqa: E402
    build_index,
    get_vectorstore,
    index_path,
)


def main() -> int:
    settings = get_settings()
    print(f"Provider: {settings.llm_provider}")

    documents = load_corpus()
    per_doc = Counter(doc.metadata["doc_id"] for doc in documents)

    print(f"\nLoaded {len(per_doc)} documents, {len(documents)} chunks:")
    for doc_id, count in sorted(per_doc.items()):
        sample = next(d for d in documents if d.metadata["doc_id"] == doc_id)
        roles = sample.metadata["allowed_roles"]
        print(f"  {doc_id:<16} {count:>2} chunks  roles: {roles}")

    print("\nEmbedding and indexing...")
    build_index(documents, settings=settings)
    print(f"Index written to {index_path(settings)}/")

    query = "advanced imaging lumbar spine conservative therapy"
    print(f"\nSanity check — top matches for {query!r} (lower score = closer):")
    store = get_vectorstore(settings)
    for doc, score in store.similarity_search_with_score(query, k=3):
        print(
            f"  {score:.3f}  {doc.metadata['doc_id']} — {doc.metadata['section'][:48]}"
        )

    print("\nAccess metadata check — same query, filtered by role:")
    for role in ("member_services", "claims_analyst", "clinician"):
        hits = store.similarity_search(
            "behavioral health outpatient therapy authorization",
            k=3,
            filter={f"role_{role}": True},
        )
        found = ", ".join(sorted({hit.metadata["doc_id"] for hit in hits}))
        print(f"  {role:<18} sees: {found}")
    print("  (POL-BEH-002 is restricted — only clinician and compliance see it)")

    if settings.llm_provider == "mock":
        print(
            "\nNote: mock embeddings match on shared words, not meaning. "
            "Run without LLM_PROVIDER=mock for real semantic ranking."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
