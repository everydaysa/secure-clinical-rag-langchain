"""The LangGraph workflow: redact, retrieve, generate, verify.

Each stage is a node with one job, and the edges make the control flow
explicit rather than buried in a function. That structure is the point: you
can show an auditor the graph and say which node enforces which control.

    redact ──> retrieve ──┬─(nothing permitted)─> refuse ──┐
                          └─(context found)─────> generate ─┴─> verify ──> audit

Generation is constrained to the retrieved context. If the context does not
support an answer, the model is instructed to emit INSUFFICIENT_CONTEXT, and
the refusal is produced by code rather than improvised by the model.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .audit import AuditRecord, digest, write_record
from .config import Settings, get_settings
from .llm import get_chat_model
from .phi import redact, scan
from .retrieval import Principal, format_context, retrieve

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = """You answer questions about health plan policy for internal staff.

Rules:
1. Use ONLY the numbered context below. Do not use outside knowledge.
2. Cite the document for every factual claim, in square brackets, exactly as
   the context labels it — for example [POL-IMG-004 v3.2].
3. Questions often mention a specific member, whose identifiers have been
   replaced with placeholders such as <PERSON> and <DATE_OF_BIRTH>. Answer with
   the policy rule that applies and the criteria that would decide the case.
   State what documentation is required. Never issue an individual approval or
   denial, and never ask for the redacted identifiers.
4. Reply with exactly INSUFFICIENT_CONTEXT only when the context contains no
   applicable rule. Missing member-specific facts is not a reason to refuse:
   state the rule and the conditions instead.
5. Do not include patient identifiers in your answer.
6. Be brief and specific. Quote thresholds and timeframes exactly.

Context:
{context}"""

REFUSAL_NO_ACCESS = (
    "No policy documents available to your role match this question. "
    "If you believe you should have access, contact the policy owner."
)
REFUSAL_NO_BASIS = (
    "The available policy documents do not answer this question. "
    "No answer was generated, to avoid guessing."
)

CITATION = re.compile(r"\[([A-Z]+-[A-Z]+-\d+)[^\]]*\]")


class GraphState(TypedDict, total=False):
    """Everything that flows through the workflow."""

    question: str
    principal: Principal
    redacted_question: str
    search_query: str
    input_phi: dict[str, int]
    documents: list[Any]
    citations: list[str]
    answer: str
    model_output: str
    refused: bool
    refusal_reason: str
    output_phi: dict[str, int]
    uncited_claims: bool
    started_at: float
    audit_event_id: str
    settings: Settings


def _settings(state: GraphState) -> Settings:
    return state.get("settings") or get_settings()


def redact_node(state: GraphState) -> GraphState:
    """Strip patient identifiers before anything leaves the process."""
    result = redact(state["question"])
    return {"redacted_question": result.text, "input_phi": result.entity_counts}


PLACEHOLDER = re.compile(r"<[A-Z_]+>")


def search_text(redacted_question: str) -> str:
    """Strip redaction placeholders before embedding the query.

    The model should still see "<PERSON>" — it explains why the question
    mentions someone — but the vector search should not. Placeholders carry no
    clinical meaning and measurably drag the match toward document headers
    instead of the section that answers the question.
    """
    text = PLACEHOLDER.sub(" ", redacted_question)
    text = re.sub(r"[ ,;:]{2,}", " ", text)  # tidy the punctuation left behind
    text = text.strip(" ,;:")

    # If the question was mostly identifiers, there is nothing left to search
    # on. Fall back to the redacted form rather than querying on fragments.
    return text if len(text.split()) >= 3 else redacted_question


def retrieve_node(state: GraphState) -> GraphState:
    """Search the index as the calling principal."""
    query = search_text(state["redacted_question"])
    result = retrieve(
        state["principal"],
        query,
        settings=_settings(state),
    )
    return {
        "search_query": query,
        "documents": result.documents,
        "citations": result.citations(),
    }


def route_after_retrieval(state: GraphState) -> Literal["generate", "refuse"]:
    return "generate" if state.get("documents") else "refuse"


def generate_node(state: GraphState) -> GraphState:
    """Answer strictly from the retrieved context."""
    from .retrieval import RetrievalResult

    context = format_context(RetrievalResult(documents=state["documents"]))
    model = get_chat_model(_settings(state))

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=state["redacted_question"]),
        ]
    )
    answer = str(response.content).strip()

    if INSUFFICIENT in answer:
        return {
            "answer": REFUSAL_NO_BASIS,
            "model_output": answer,
            "refused": True,
            "refusal_reason": "insufficient_context",
        }
    return {"answer": answer, "model_output": answer, "refused": False}


def refuse_node(state: GraphState) -> GraphState:
    """No documents the caller may see — refuse without calling the model."""
    return {
        "answer": REFUSAL_NO_ACCESS,
        "refused": True,
        "refusal_reason": "no_permitted_documents",
        "citations": [],
    }


def dedupe_citations(answer: str) -> str:
    """Collapse runs of the same citation: "[X] [X] [X]" -> "[X]"."""
    return re.sub(r"(\[[^\]]+\])(\s*\1)+", r"\1", answer)


def verify_node(state: GraphState) -> GraphState:
    """Check the answer for leaked identifiers and unsupported citations."""
    answer = dedupe_citations(state.get("answer", ""))
    output_phi = scan(answer, use_ner=False)

    permitted = {doc.metadata["doc_id"] for doc in state.get("documents", [])}
    cited = set(CITATION.findall(answer))

    # A citation naming a document that was never retrieved is a fabrication.
    invented = cited - permitted
    if invented:
        return {
            "answer": REFUSAL_NO_BASIS,
            "refused": True,
            "refusal_reason": f"citation_not_in_context:{','.join(sorted(invented))}",
            "output_phi": output_phi,
            "uncited_claims": True,
        }

    return {
        "answer": answer,
        "output_phi": output_phi,
        "uncited_claims": not cited and not state.get("refused", False),
    }


def _document_summary(documents: list[Any]) -> list[str]:
    """One entry per document, with how many of its chunks were used."""
    counts: dict[str, int] = {}
    for doc in documents:
        tag = f"{doc.metadata['doc_id']} v{doc.metadata['version']}"
        counts[tag] = counts.get(tag, 0) + 1
    return [
        tag if count == 1 else f"{tag} x{count}" for tag, count in counts.items()
    ]


def audit_node(state: GraphState) -> GraphState:
    """Append one PHI-free record describing what happened."""
    settings = _settings(state)
    principal = state["principal"]
    started = state.get("started_at") or time.monotonic()

    record = AuditRecord(
        user_id=principal.user_id,
        role=principal.role,
        question_hash=digest(state["question"]),
        question_redacted=state.get("redacted_question", ""),
        input_phi=state.get("input_phi", {}),
        documents=_document_summary(state.get("documents", [])),
        refused=bool(state.get("refused", False)),
        refusal_reason=state.get("refusal_reason", ""),
        answer_hash=digest(state.get("answer", "")),
        output_phi=state.get("output_phi", {}),
        uncited_claims=bool(state.get("uncited_claims", False)),
        provider=settings.llm_provider,
        chat_deployment=settings.azure_openai_chat_deployment,
        latency_ms=int((time.monotonic() - started) * 1000),
    )
    write_record(record, settings.audit_log_path)
    return {"audit_event_id": record.event_id}


def build_graph():
    """Compile the workflow."""
    graph = StateGraph(GraphState)

    graph.add_node("redact", redact_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("verify", verify_node)
    graph.add_node("audit", audit_node)

    graph.add_edge(START, "redact")
    graph.add_edge("redact", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieval)
    graph.add_edge("generate", "verify")
    graph.add_edge("refuse", "verify")
    graph.add_edge("verify", "audit")
    graph.add_edge("audit", END)

    return graph.compile()


def ask(
    question: str,
    principal: Principal,
    settings: Settings | None = None,
) -> GraphState:
    """Run one question through the workflow and return the final state."""
    app = build_graph()
    return app.invoke(
        {
            "question": question,
            "principal": principal,
            "started_at": time.monotonic(),
            "settings": settings or get_settings(),
        }
    )
