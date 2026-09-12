"""Access control: the tests that would fail loudly if entitlement broke."""

from __future__ import annotations

import pytest

from clinical_rag.access import UnknownRoleError, access_metadata, validate_role
from clinical_rag.config import Settings
from clinical_rag.retrieval import Principal, retrieve

RESTRICTED = "POL-BEH-002"
RESTRICTED_QUESTION = (
    "How many outpatient behavioral health therapy visits before prior authorization?"
)


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(UnknownRoleError):
        validate_role("admin")


def test_principal_rejects_unknown_role() -> None:
    with pytest.raises(UnknownRoleError):
        Principal(user_id="attacker", role="superuser")


def test_access_metadata_expands_every_role() -> None:
    flags = access_metadata(["clinician", "compliance_auditor"])
    assert flags["role_clinician"] is True
    assert flags["role_claims_analyst"] is False
    assert len(flags) == 4  # one flag per known role, not just the allowed ones


def test_document_with_unknown_role_is_rejected() -> None:
    with pytest.raises(UnknownRoleError):
        access_metadata(["clinician", "not_a_role"])


@pytest.mark.parametrize("role", ["claims_analyst", "member_services"])
def test_restricted_document_is_never_retrieved(role: str, indexed: Settings) -> None:
    result = retrieve(
        Principal(user_id="t", role=role), RESTRICTED_QUESTION, k=8, settings=indexed
    )
    assert all(doc.metadata["doc_id"] != RESTRICTED for doc in result.documents)


@pytest.mark.parametrize("role", ["clinician", "compliance_auditor"])
def test_entitled_roles_do_retrieve_it(role: str, indexed: Settings) -> None:
    result = retrieve(
        Principal(user_id="t", role=role), RESTRICTED_QUESTION, k=8, settings=indexed
    )
    assert any(doc.metadata["doc_id"] == RESTRICTED for doc in result.documents)


def test_every_chunk_carries_the_callers_flag(indexed: Settings) -> None:
    """Defence in depth: whatever comes back must be provably permitted."""
    result = retrieve(
        Principal(user_id="t", role="member_services"),
        "claims and appeals",
        k=8,
        settings=indexed,
    )
    assert result.documents
    assert all(doc.metadata["role_member_services"] is True for doc in result.documents)
    assert result.withheld == 0
