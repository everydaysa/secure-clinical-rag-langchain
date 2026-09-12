"""Audit logging.

One JSON line per question. The design constraint that shapes everything here:
**an audit trail must be safe to keep.** A log that records what a user asked,
verbatim, becomes a second copy of the PHI you just worked to remove — usually
in a system with weaker access controls than the source.

So each record holds:

* who asked, under which role, and when
* a salted hash of the original question, which lets an investigator confirm
  "was this exact question asked?" without the log itself revealing it
* the redacted question, which has already had identifiers removed
* which documents were retrieved, and which were withheld
* what the PHI detectors found, as types and counts only
* whether the answer was refused, and why
* a hash of the answer, so an answer shown to a user can be tied to a record

It does not hold: raw questions, raw answers, or any identifier value.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A salt keeps the hashes from being reversible by guessing common questions.
# In production this belongs in a secrets store, not an environment default.
HASH_SALT = os.environ.get("AUDIT_HASH_SALT", "local-dev-salt")


def digest(text: str) -> str:
    """Salted SHA-256, truncated — enough to match, useless to reverse."""
    return hashlib.sha256(f"{HASH_SALT}:{text}".encode("utf-8")).hexdigest()[:32]


@dataclass
class AuditRecord:
    """One question, as it is safe to record it."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    user_id: str = ""
    role: str = ""
    question_hash: str = ""
    question_redacted: str = ""
    input_phi: dict[str, int] = field(default_factory=dict)
    documents: list[str] = field(default_factory=list)
    withheld: int = 0
    refused: bool = False
    refusal_reason: str = ""
    answer_hash: str = ""
    output_phi: dict[str, int] = field(default_factory=dict)
    uncited_claims: bool = False
    provider: str = ""
    chat_deployment: str = ""
    latency_ms: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def write_record(record: AuditRecord, path: str | Path) -> Path:
    """Append one record as a line of JSON, creating the log if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json() + "\n")
    return path


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read the log back, for review scripts and tests."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
