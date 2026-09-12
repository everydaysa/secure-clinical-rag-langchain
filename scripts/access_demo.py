"""Show what each role can and cannot retrieve.

Run:  python scripts/access_demo.py
      LLM_PROVIDER=mock python scripts/access_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.access import ROLES, UnknownRoleError  # noqa: E402
from clinical_rag.retrieval import Principal, retrieve  # noqa: E402

QUESTIONS = [
    "How many outpatient therapy visits are allowed before prior authorization?",
    "When is an MRI covered for low back pain?",
    "How long do I have to file an appeal?",
]


def main() -> int:
    for question in QUESTIONS:
        print(f"Q: {question}")
        for role in ROLES:
            principal = Principal(user_id=f"demo.{role}", role=role)
            result = retrieve(principal, question, k=3)
            if result.empty:
                print(f"   {role:<18} -> no permitted documents (request refused)")
            else:
                print(f"   {role:<18} -> {', '.join(result.citations())}")
        print()

    print("Deny by default — an unrecognised role is rejected, not defaulted:")
    try:
        Principal(user_id="attacker", role="admin")
    except UnknownRoleError as exc:
        print(f"   {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
