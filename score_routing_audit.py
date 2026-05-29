"""Score a manually labeled topic-routing audit JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a filled routing audit file.")
    parser.add_argument("--audit", type=Path, default=Path("results/routing_audit.jsonl"))
    parser.add_argument(
        "--show-misses",
        type=int,
        default=20,
        help="Maximum number of mislabeled rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.audit)
    labeled = [row for row in rows if row.get("correct_topic")]
    if not labeled:
        raise SystemExit("No labeled rows found. Fill in correct_topic first.")

    correct = [
        row for row in labeled
        if row["predicted_topic"].strip() == row["correct_topic"].strip()
    ]
    accuracy = len(correct) / len(labeled) * 100
    print(f"Routing accuracy: {len(correct)} / {len(labeled)} ({accuracy:.2f}%)")

    predicted_counts = Counter(row["predicted_topic"] for row in labeled)
    gold_counts = Counter(row["correct_topic"] for row in labeled)
    print("\nPredicted topic counts:")
    for topic, count in sorted(predicted_counts.items()):
        print(f"  {topic}: {count}")

    print("\nCorrect topic counts:")
    for topic, count in sorted(gold_counts.items()):
        print(f"  {topic}: {count}")

    misses = [row for row in labeled if row not in correct]
    if misses:
        print("\nMisses:")
        for row in misses[: args.show_misses]:
            print(
                f"  id={row.get('id')} predicted={row['predicted_topic']} "
                f"correct={row['correct_topic']} notes={row.get('notes', '')}"
            )
            print(f"    {row.get('question', '')}")


if __name__ == "__main__":
    main()

