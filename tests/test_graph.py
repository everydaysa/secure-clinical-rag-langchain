"""The workflow: routing, output verification, and what reaches the audit log."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from clinical_rag.config import Settings
from clinical_rag.graph import (
    REFUSAL_NO_ACCESS,
    REFUSAL_NO_BASIS,
    ask,
    dedupe_citations,
    route_after_retrieval,
    search_text,
    verify_node,
)
from clinical_rag.retrieval import Principal

DOC = Document(page_content="x", metadata={"doc_id": "POL-IMG-004", "version": "3.2"})


def test_search_text_strips_placeholders() -> None:
    cleaned = search_text("Is an MRI covered for <PERSON>, <DATE_OF_BIRTH>, after PT?")
    assert "<" not in cleaned
    assert "MRI" in cleaned and "PT" in cleaned


def test_search_text_falls_back_when_nothing_is_left() -> None:
    original = "<MEMBER_ID> <CLAIM_ID>"
    assert search_text(original) == original


def test_routing_refuses_when_nothing_permitted() -> None:
    assert route_after_retrieval({"documents": []}) == "refuse"
    assert route_after_retrieval({"documents": [DOC]}) == "generate"


def test_fabricated_citation_is_refused() -> None:
    state = verify_node({"answer": "Covered [POL-XYZ-999 v1.0].", "documents": [DOC]})
    assert state["refused"] is True
    assert "POL-XYZ-999" in state["refusal_reason"]
    assert state["answer"] == REFUSAL_NO_BASIS


def test_valid_citation_passes() -> None:
    state = verify_node({"answer": "Covered [POL-IMG-004 v3.2].", "documents": [DOC]})
    assert state.get("refused") is not True
    assert state["uncited_claims"] is False


def test_missing_citation_is_flagged() -> None:
    state = verify_node({"answer": "Covered after six weeks.", "documents": [DOC]})
    assert state["uncited_claims"] is True


def test_identifier_in_answer_is_detected() -> None:
    state = verify_node(
        {"answer": "Call 615-555-0142 [POL-IMG-004 v3.2].", "documents": [DOC]}
    )
    assert state["output_phi"] == {"PHONE_NUMBER": 1}


def test_repeated_citations_are_collapsed() -> None:
    answer = "Rule applies. [POL-IMG-004 v3.2] [POL-IMG-004 v3.2] [POL-IMG-004 v3.2]"
    assert dedupe_citations(answer).count("[POL-IMG-004") == 1


def test_no_permitted_documents_refuses_without_calling_the_model(
    indexed: Settings, monkeypatch
) -> None:
    from clinical_rag import graph as graph_module
    from clinical_rag.retrieval import RetrievalResult

    monkeypatch.setattr(
        graph_module, "retrieve", lambda *a, **k: RetrievalResult(documents=[])
    )

    state = ask("anything at all", Principal(user_id="t", role="clinician"), indexed)
    assert state["refused"] is True
    assert state["refusal_reason"] == "no_permitted_documents"
    assert state["answer"] == REFUSAL_NO_ACCESS


def test_audit_record_contains_no_phi(indexed: Settings) -> None:
    question = "Is an MRI covered for John Smith, DOB 3/4/1961, MRN 4471923?"
    state = ask(question, Principal(user_id="t.tester", role="claims_analyst"), indexed)

    raw = Path(indexed.audit_log_path).read_text(encoding="utf-8")
    assert state["audit_event_id"] in raw

    record = json.loads(raw.strip().splitlines()[-1])
    serialised = json.dumps(record)

    for secret in ("John", "Smith", "4471923", "3/4/1961"):
        assert secret not in serialised

    assert record["role"] == "claims_analyst"
    assert record["input_phi"]["MRN"] == 1
    assert record["question_hash"] and record["question_hash"] != question
