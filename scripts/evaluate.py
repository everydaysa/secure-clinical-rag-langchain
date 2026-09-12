"""Run the evaluation set and report pass/fail per check.

Run:  python scripts/evaluate.py            # against the configured provider
      LLM_PROVIDER=mock python scripts/evaluate.py   # retrieval checks only

Exit code is non-zero if any check fails, so this can gate a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.config import get_settings  # noqa: E402
from clinical_rag.graph import ask  # noqa: E402
from clinical_rag.retrieval import Principal  # noqa: E402

CASES = Path("evals/cases.yaml")

# Checks that depend on the language model rather than on retrieval.
MODEL_CHECKS = {"refuses", "contains", "cites"}


@dataclass
class Check:
    case: str
    name: str
    status: str  # pass | fail | skip
    detail: str = ""


def evaluate_case(case: dict, model_available: bool) -> list[Check]:
    principal = Principal(user_id="eval", role=case["role"])
    state = ask(case["question"], principal)

    retrieved = {doc.metadata["doc_id"] for doc in state.get("documents", [])}
    answer = state.get("answer", "")
    cited = state.get("citations", [])
    checks: list[Check] = []

    for expected in case.get("retrieves", []):
        ok = expected in retrieved
        checks.append(
            Check(case["id"], f"retrieves {expected}", "pass" if ok else "fail",
                  "" if ok else f"got {sorted(retrieved) or 'nothing'}")
        )

    for forbidden in case.get("forbids", []):
        ok = forbidden not in retrieved
        checks.append(
            Check(case["id"], f"never retrieves {forbidden}", "pass" if ok else "fail",
                  "" if ok else "ACCESS CONTROL FAILURE")
        )

    for entity in case.get("phi", []):
        ok = entity in state.get("input_phi", {})
        checks.append(
            Check(case["id"], f"redacts {entity}", "pass" if ok else "fail",
                  "" if ok else f"got {state.get('input_phi')}")
        )

    # Output must never contain identifiers, whatever the case says.
    leaked = state.get("output_phi") or {}
    checks.append(
        Check(case["id"], "no identifiers in answer", "pass" if not leaked else "fail",
              "" if not leaked else str(leaked))
    )

    if not model_available:
        for name in MODEL_CHECKS:
            if name in case:
                checks.append(Check(case["id"], name, "skip", "mock provider"))
        return checks

    if "refuses" in case:
        ok = bool(state.get("refused", False)) == bool(case["refuses"])
        checks.append(
            Check(case["id"], f"refuses={case['refuses']}", "pass" if ok else "fail",
                  "" if ok else f"refused={state.get('refused')} "
                                f"{state.get('refusal_reason', '')}")
        )

    for phrase in case.get("contains", []):
        ok = phrase.lower() in answer.lower()
        checks.append(
            Check(case["id"], f"answer contains {phrase!r}", "pass" if ok else "fail",
                  "" if ok else answer[:80])
        )

    for doc_id in case.get("cites", []):
        ok = any(doc_id in citation for citation in cited) and doc_id in answer
        checks.append(
            Check(case["id"], f"cites {doc_id}", "pass" if ok else "fail",
                  "" if ok else f"citations={cited}")
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation set")
    parser.add_argument("--cases", default=str(CASES))
    args = parser.parse_args()

    settings = get_settings()
    model_available = settings.llm_provider != "mock"

    cases = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))
    print(f"Provider: {settings.llm_provider}")
    if not model_available:
        print("Answer-quality checks are skipped: the mock model returns a fixed reply.")
    print()

    results: list[Check] = []
    for case in cases:
        results.extend(evaluate_case(case, model_available))

    width = max(len(check.name) for check in results) + 2
    current = ""
    for check in results:
        if check.case != current:
            current = check.case
            print(f"{current}")
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}[check.status]
        detail = f"  {check.detail}" if check.detail else ""
        print(f"   [{mark}] {check.name:<{width}}{detail}")

    passed = sum(1 for c in results if c.status == "pass")
    failed = sum(1 for c in results if c.status == "fail")
    skipped = sum(1 for c in results if c.status == "skip")

    print()
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
