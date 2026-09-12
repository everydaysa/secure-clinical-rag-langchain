"""Role-filtered retrieval.

The caller is represented by a :class:`Principal` — an identity plus a single
role. Every search is filtered by that role at the vector store, so documents
the caller is not entitled to never enter the process, let alone the prompt.

Three properties worth naming, because they are the difference between a
control and a hint:

* **Deny by default.** An unknown role raises; it does not fall back to "show
  everything" or "show public only".
* **Filter at the source.** The ``where`` clause runs inside Chroma. Fetching
  everything and filtering afterwards would still pull restricted text into
  application memory.
* **Verify the result.** Returned chunks are re-checked against the caller's
  role. If the store ever returns something it shouldn't, the chunk is dropped
  and the event is surfaced rather than silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from .access import ROLE_DESCRIPTIONS, role_flag, validate_role
from .config import Settings, get_settings
from .vectorstore import get_vectorstore

DEFAULT_K = 5


@dataclass(frozen=True)
class Principal:
    """Who is asking, and under which role."""

    user_id: str
    role: str

    def __post_init__(self) -> None:
        validate_role(self.role)

    @property
    def description(self) -> str:
        return ROLE_DESCRIPTIONS[self.role]


class AccessViolation(RuntimeError):
    """Raised when the store returns a document the principal may not see."""


@dataclass
class RetrievalResult:
    """Documents the principal is allowed to see, plus what was withheld."""

    documents: list[Document]
    withheld: int = 0

    @property
    def empty(self) -> bool:
        return not self.documents

    def citations(self) -> list[str]:
        seen: list[str] = []
        for doc in self.documents:
            tag = f"{doc.metadata['doc_id']} v{doc.metadata['version']}"
            if tag not in seen:
                seen.append(tag)
        return seen


def retrieve(
    principal: Principal,
    query: str,
    k: int = DEFAULT_K,
    settings: Settings | None = None,
    strict: bool = False,
) -> RetrievalResult:
    """Search the index as ``principal``, returning only permitted chunks."""
    settings = settings or get_settings()
    flag = role_flag(principal.role)

    store = get_vectorstore(settings)
    hits = store.similarity_search(query, k=k, filter={flag: True})

    permitted: list[Document] = []
    withheld = 0
    for doc in hits:
        if doc.metadata.get(flag) is True:
            permitted.append(doc)
            continue
        withheld += 1
        if strict:
            raise AccessViolation(
                f"{doc.metadata.get('doc_id')} returned for role {principal.role}"
            )

    return RetrievalResult(documents=permitted, withheld=withheld)


def format_context(result: RetrievalResult, max_chars: int = 4000) -> str:
    """Render retrieved chunks as numbered, citable context for the prompt."""
    blocks: list[str] = []
    used = 0
    for index, doc in enumerate(result.documents, start=1):
        meta = doc.metadata
        header = (
            f"[{index}] {meta['doc_id']} v{meta['version']} — {meta['title']} — "
            f"{meta['section']}"
        )
        block = f"{header}\n{doc.page_content.strip()}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
