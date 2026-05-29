"""Generate a small manual audit file for topic routing quality.

The output JSONL is meant to be edited by hand: fill in ``correct_topic`` and
``notes`` for each row, then use those misses to improve ``prompt_strategy.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_strategy import classify_math_topic


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def compact(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a topic-routing audit sample.")
    parser.add_argument("--data", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/routing_audit.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-question-chars", type=int, default=500)
    parser.add_argument(
        "--topic",
        default=None,
        help="Only audit examples currently routed to this topic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.data)
    rows = []

    for index, item in enumerate(data[args.offset:], start=args.offset):
        topic = classify_math_topic(item["question"]).name
        if args.topic and topic != args.topic:
            continue

        rows.append(
            {
                "index": index,
                "id": item.get("id"),
                "predicted_topic": topic,
                "correct_topic": "",
                "is_mcq": bool(item.get("options")),
                "question": compact(item["question"], args.max_question_chars),
                "notes": "",
            }
        )

        if len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit("No audit rows matched the requested filters.")

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} routing audit rows to {args.output}")
    print("Fill in correct_topic and notes for mistakes, then we can tune prompt_strategy.py.")


if __name__ == "__main__":
    main()

