"""Summarise the audit log — the access review an auditor would ask for.

Run:  python scripts/audit_review.py
      python scripts/audit_review.py --restricted POL-BEH-002
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_rag.audit import read_records  # noqa: E402
from clinical_rag.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Review the audit log")
    parser.add_argument("--log", default=None, help="path to audit.jsonl")
    parser.add_argument(
        "--restricted",
        default="POL-BEH-002",
        help="document to list access events for",
    )
    args = parser.parse_args()

    path = args.log or get_settings().audit_log_path
    records = read_records(path)
    if not records:
        print(f"No audit records at {path}. Ask some questions first.")
        return 1

    print(f"Audit log: {path}")
    print(f"Records  : {len(records)}")
    print(f"Window   : {records[0]['timestamp']} .. {records[-1]['timestamp']}")
    print()

    by_role = Counter(record["role"] for record in records)
    print("Questions by role")
    for role, count in by_role.most_common():
        refused = sum(
            1 for r in records if r["role"] == role and r["refused"]
        )
        print(f"  {role:<20} {count:>3} asked, {refused:>3} refused")
    print()

    reasons = Counter(r["refusal_reason"] for r in records if r["refused"])
    if reasons:
        print("Refusal reasons")
        for reason, count in reasons.most_common():
            print(f"  {reason:<40} {count}")
        print()

    phi = Counter()
    for record in records:
        phi.update(record["input_phi"])
    print("PHI detected in questions (types and counts only)")
    if phi:
        for entity, count in phi.most_common():
            print(f"  {entity:<20} {count}")
    else:
        print("  none")
    print()

    leaks = [r for r in records if r["output_phi"]]
    print(f"Answers with identifiers detected: {len(leaks)}")
    uncited = [r for r in records if r["uncited_claims"]]
    print(f"Answers with no citation          : {len(uncited)}")
    print()

    print(f"Access events for {args.restricted}")
    hits = [
        record
        for record in records
        if any(doc.startswith(args.restricted) for doc in record["documents"])
    ]
    if not hits:
        print("  none")
    for record in hits:
        print(
            f"  {record['timestamp']}  {record['user_id']:<18} "
            f"{record['role']:<20} event {record['event_id'][:8]}"
        )
    print()

    latencies = sorted(record["latency_ms"] for record in records)
    median = latencies[len(latencies) // 2]
    print(f"Latency: median {median} ms, max {latencies[-1]} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
