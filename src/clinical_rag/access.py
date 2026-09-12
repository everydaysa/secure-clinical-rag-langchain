"""Role definitions and the access metadata stored alongside each chunk.

Access control lives in the index, not in the prompt. Every chunk carries one
boolean flag per role, so the vector store can filter documents a caller is not
entitled to see *before* anything reaches the model. Asking the model nicely to
withhold content is not a control; a metadata filter is.
"""

from __future__ import annotations

ROLES: tuple[str, ...] = (
    "member_services",
    "claims_analyst",
    "clinician",
    "compliance_auditor",
)

ROLE_DESCRIPTIONS: dict[str, str] = {
    "member_services": "Answers member calls. Sees member-facing material only.",
    "claims_analyst": "Adjudicates claims. Sees payer policy, not restricted clinical policy.",
    "clinician": "Clinical reviewer. Sees clinical guidance and restricted policy.",
    "compliance_auditor": "Audits decisions. Sees everything, read-only.",
}

FLAG_PREFIX = "role_"


class UnknownRoleError(ValueError):
    """Raised when a caller presents a role the system does not define."""


def validate_role(role: str) -> str:
    """Return the role if it is known, otherwise raise (deny by default)."""
    if role not in ROLES:
        raise UnknownRoleError(
            f"Unknown role {role!r}. Known roles: {', '.join(ROLES)}"
        )
    return role


def role_flag(role: str) -> str:
    """Metadata key holding the access flag for a role."""
    return f"{FLAG_PREFIX}{validate_role(role)}"


def access_metadata(allowed_roles: list[str]) -> dict[str, bool]:
    """Expand a document's allowed_roles list into one boolean per known role.

    Chroma metadata values must be scalars, so a list cannot be filtered on
    directly. One flag per role keeps the filter server-side and explicit.
    Unknown roles in the source document are rejected rather than ignored.
    """
    for role in allowed_roles:
        validate_role(role)
    return {role_flag(role): role in allowed_roles for role in ROLES}
