from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT = "data/public_run_1.jsonl"


def default_output_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_clean.jsonl")


def nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and any(
        (nonempty_str(item) if isinstance(item, str) else bool(item))
        for item in value
    )


def clean_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    if record.get("error"):
        return False, "error"

    if record.get("status") != "correct":
        return False, f"status:{record.get('status')}"

    if record.get("answer_match") is not True:
        return False, f"answer_match:{record.get('answer_match')}"

    if not nonempty_list(record.get("generated_traces")):
        return False, "no_generated_traces"

    if not nonempty_str(record.get("predicted_answer")):
        return False, "no_predicted_answer"

    if record.get("expected_answer") in (None, "", []):
        return False, "no_expected_answer"

    extracted_answers = record.get("extracted_answers")
    if not nonempty_list(extracted_answers):
        return False, "no_extracted_answers"

    trace_parse_statuses = record.get("trace_parse_statuses")
    if isinstance(trace_parse_statuses, list) and "ok" not in trace_parse_statuses:
        return False, "no_ok_trace"

    return True, "kept"


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, str(exc)


def clean_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    removal_reasons: Counter[str] = Counter()
    input_rows = 0
    invalid_json = 0

    for _line_no, record, parse_error in iter_jsonl(input_path):
        if parse_error is not None:
            invalid_json += 1
            removal_reasons["invalid_json"] += 1
            continue

        assert record is not None
        input_rows += 1
        should_keep, reason = clean_record(record)
        if should_keep:
            kept.append(record)
        else:
            removal_reasons[reason] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": input_rows,
        "invalid_json": invalid_json,
        "kept_rows": len(kept),
        "removed_rows": input_rows + invalid_json - len(kept),
        "removal_reasons": dict(removal_reasons),
    }


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only public teacher trace rows with correct judged answers."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file. Defaults to '<input_stem>_clean.jsonl' next to input.",
    )
    return parser


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_for(input_path)
    summary = clean_file(input_path, output_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
