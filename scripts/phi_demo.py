"""Show the PHI redaction layer on realistic (synthetic) questions.

Run:  python scripts/phi_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.phi import redact  # noqa: E402

EXAMPLES = [
    "Is an MRI covered for John Smith, DOB 3/4/1961, with 8 weeks of PT?",
    "Member ID BCX449281 called about claim #100238845 denied on 2026-08-14.",
    "Patient MRN 4471923, ssn 123-45-6789, phone 615-555-0142, "
    "email j.smith@example.com — needs CPAP supplies.",
    "What is the appeal window for a denied specialty drug?",
    "Jane Doe in Nashville had 6 weeks of conservative therapy since Jan 5, 2026.",
]


def main() -> int:
    for question in EXAMPLES:
        result = redact(question)
        print("in : ", question)
        print("out: ", result.text)
        print("phi: ", result.summary())
        print()

    print("Note: only entity types and counts are recorded for audit —")
    print("the values themselves are never written to logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
