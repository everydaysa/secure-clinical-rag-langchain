"""Ask the assistant a question as a given role.

Examples:
  python scripts/ask.py --role claims_analyst "When is an MRI covered for low back pain?"
  python scripts/ask.py --role member_services "How many therapy visits before prior auth?"
  LLM_PROVIDER=mock python scripts/ask.py --role clinician "CPAP adherence rule?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.access import ROLES  # noqa: E402
from clinical_rag.graph import ask  # noqa: E402
from clinical_rag.retrieval import Principal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the clinical policy assistant")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--role", choices=ROLES, default="claims_analyst")
    parser.add_argument("--user", default="demo.user")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print the retrieved chunks that were sent to the model",
    )
    args = parser.parse_args()

    principal = Principal(user_id=args.user, role=args.role)
    state = ask(args.question, principal)

    print(f"Asked as : {principal.user_id} ({principal.role})")
    print(f"Question : {args.question}")
    if state.get("input_phi"):
        counts = ", ".join(f"{k}x{v}" for k, v in sorted(state["input_phi"].items()))
        print(f"Redacted : {state['redacted_question']}")
        print(f"PHI found: {counts}")
    if args.show_context:
        print(f"Search   : {state.get('search_query', '')}")
        print("Retrieved:")
        for doc in state.get("documents", []):
            meta = doc.metadata
            print(f"   {meta['doc_id']} — {meta['section'][:60]}")
        if state.get("model_output"):
            print(f"Raw model: {state['model_output'][:200]}")
    print()

    print(state["answer"])
    print()

    if state.get("citations"):
        print(f"Sources  : {', '.join(state['citations'])}")
    if state.get("refused"):
        print(f"Refused  : {state.get('refusal_reason')}")
    if state.get("output_phi"):
        print(f"WARNING  : identifiers detected in output: {state['output_phi']}")
    if state.get("uncited_claims"):
        print("WARNING  : answer contained no citation")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
