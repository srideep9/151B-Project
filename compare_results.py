"""Compare two quick_eval JSONL result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path: Path) -> dict:
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["id"]: row for row in rows}


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(row["correct"] for row in rows) / len(rows) * 100


def summarize(label: str, rows: list[dict]) -> str:
    correct = sum(row["correct"] for row in rows)
    total = len(rows)
    return f"{label}: {correct}/{total} ({accuracy(rows):.2f}%)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and routed quick_eval outputs.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--show", type=int, default=20, help="Maximum changed rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_results(args.base)
    candidate = load_results(args.candidate)
    shared_ids = sorted(set(base) & set(candidate))
    if not shared_ids:
        raise SystemExit("No overlapping ids found between result files.")

    base_rows = [base[item_id] for item_id in shared_ids]
    candidate_rows = [candidate[item_id] for item_id in shared_ids]
    wins = [item_id for item_id in shared_ids if not base[item_id]["correct"] and candidate[item_id]["correct"]]
    losses = [item_id for item_id in shared_ids if base[item_id]["correct"] and not candidate[item_id]["correct"]]
    ties = [item_id for item_id in shared_ids if base[item_id]["correct"] == candidate[item_id]["correct"]]

    print(f"Compared {len(shared_ids)} shared examples")
    print(summarize("Base", base_rows))
    print(summarize("Candidate", candidate_rows))
    print(f"Wins: {len(wins)}  Losses: {len(losses)}  Ties: {len(ties)}")

    topics = sorted({candidate[item_id].get("topic", "unknown") for item_id in shared_ids})
    if topics:
        print("\nBy candidate topic:")
        for topic in topics:
            topic_ids = [item_id for item_id in shared_ids if candidate[item_id].get("topic", "unknown") == topic]
            b_rows = [base[item_id] for item_id in topic_ids]
            c_rows = [candidate[item_id] for item_id in topic_ids]
            print(f"  {topic}: base {accuracy(b_rows):.2f}% -> candidate {accuracy(c_rows):.2f}% ({len(topic_ids)} examples)")

    changed = wins + losses
    if changed:
        print("\nChanged rows:")
        for item_id in changed[: args.show]:
            direction = "WIN" if item_id in wins else "LOSS"
            row = candidate[item_id]
            print(
                f"  {direction} id={item_id} topic={row.get('topic')} "
                f"base={base[item_id]['correct']} candidate={row['correct']}"
            )


if __name__ == "__main__":
    main()

