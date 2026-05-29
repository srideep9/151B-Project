from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Protocol

from judger import Judger


DEFAULT_INPUT = "data/tempPaste.jsonl"


class AnswerExtractor(Protocol):
    def extract_ans(self, resp_str: str) -> str:
        ...


class BoxedFallbackExtractor:
    def extract_ans(self, resp_str: str) -> str:
        think_end = resp_str.rfind("</think>")
        search_text = resp_str[think_end + len("</think>") :] if think_end >= 0 else resp_str
        boxes = [box for box in iter_boxed(search_text) if box["balanced"]]
        if boxes:
            return boxes[-1]["content"].strip()

        matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
        return matches[-1] if matches else ""


def iter_boxed(text: str) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for match in re.finditer(r"\\+boxed\{", text):
        slash_count = len(match.group(0)) - len("boxed{")
        content_start = match.end()
        depth = 1
        idx = content_start
        while idx < len(text) and depth > 0:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1

        boxes.append(
            {
                "start": match.start(),
                "end": idx if depth == 0 else None,
                "slash_count": slash_count,
                "content": text[content_start : idx - 1] if depth == 0 else "",
                "balanced": depth == 0,
            }
        )
    return boxes


def split_csv_like(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def is_single_mcq_prompt(row: dict[str, Any]) -> bool:
    prompt = row.get("system_prompt") or ""
    return "ONLY the final multiple-choice answer" in prompt


def expected_answer_count(row: dict[str, Any]) -> int | None:
    question = row.get("question") or ""
    count = question.count("[ANS]")
    return count if count > 1 else None


def check_trace(
    row: dict[str, Any],
    trace: str,
    trace_index: int,
    answer_extractor: AnswerExtractor,
    strict_format: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    stripped = trace.strip()
    think_open_count = stripped.count("<think>")
    think_close_count = stripped.count("</think>")

    if not stripped:
        failures.append("empty trace")

    if strict_format:
        if think_open_count != 1:
            failures.append(f"expected exactly one <think>, found {think_open_count}")
        if think_close_count != 1:
            failures.append(f"expected exactly one </think>, found {think_close_count}")
        if think_open_count and think_close_count and stripped.find("<think>") > stripped.find("</think>"):
            failures.append("</think> appears before <think>")
    elif think_close_count == 0:
        warnings.append("missing </think>; Judger may still extract an answer from boxed/last numeric content")

    if think_open_count and not stripped.startswith("<think>"):
        warnings.append("text appears before <think>")

    if re.match(r"(?is)^(okay|ok,|let'?s|sure|we need|i need|first,)", stripped):
        warnings.append("reasoning starts conversationally; prompt asks for concise internal reasoning")

    boxes = iter_boxed(stripped)
    balanced_boxes = [box for box in boxes if box["balanced"]]

    if strict_format and len(balanced_boxes) != 1:
        failures.append(f"expected exactly one final boxed answer, found {len(balanced_boxes)}")
    if any(not box["balanced"] for box in boxes):
        failures.append("found unbalanced boxed expression")

    if balanced_boxes:
        box = balanced_boxes[-1]
        if box["slash_count"] != 1:
            warnings.append(f"boxed command has {box['slash_count']} leading backslashes; expected 1")
        if strict_format and not box["content"].strip():
            failures.append("boxed answer is empty")

        after_box = stripped[box["end"] :].strip()
        if strict_format and after_box:
            failures.append("text appears after final boxed answer")

        think_end = stripped.rfind("</think>")
        if strict_format and think_end >= 0 and box["start"] < think_end:
            failures.append("boxed answer appears inside <think> instead of after </think>")

    else:
        box = None

    extracted = answer_extractor.extract_ans(stripped)
    if not extracted:
        failures.append("answer extractor returned empty")

    if is_single_mcq_prompt(row) and extracted and not re.fullmatch(r"[A-Z]", extracted.strip()):
        failures.append(f"MCQ extracted answer should be one uppercase letter, got {extracted!r}")

    wanted_count = expected_answer_count(row)
    if wanted_count is not None and extracted:
        got_count = len(split_csv_like(extracted))
        if got_count != wanted_count:
            warnings.append(f"question has {wanted_count} [ANS] blanks; extracted answer has {got_count} comma-separated part(s)")

    return {
        "trace_index": trace_index,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "extracted": extracted,
        "boxed": box["content"].strip() if box else "",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"id": f"line:{line_no}", "generated_traces": [], "status": "json_error", "error": str(exc)})
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether generated traces are parseable by Judger.")
    parser.add_argument("path", nargs="?", default=DEFAULT_INPUT, help=f"JSONL file to check. Default: {DEFAULT_INPUT}")
    parser.add_argument("--show-ok", action="store_true", help="Print rows/traces that pass too.")
    parser.add_argument("--max-issues", type=int, default=25, help="Maximum issue details to print.")
    parser.add_argument("--strict-format", action="store_true", help="Also require the exact prompt format: one <think> block, one final boxed answer, and no trailing text.")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.path))
    try:
        answer_extractor: AnswerExtractor = Judger(strict_extract=False)
        extractor_name = "Judger.extract_ans"
    except Exception as exc:
        answer_extractor = BoxedFallbackExtractor()
        extractor_name = f"fallback boxed extractor (Judger unavailable: {exc})"

    total_rows = len(rows)
    total_traces = 0
    parseable_traces = 0
    issue_count = 0
    rows_without_traces = 0

    print("Goal:")
    print("- default mode checks whether each trace gives a usable nonempty answer via Judger.extract_ans")
    print("- MCQ-only prompts must extract to one uppercase letter, e.g. C")
    if args.strict_format:
        print("- strict format is enabled: one <think> block, one final \\boxed{...}, no trailing text")
    else:
        print("- strict prompt formatting is not required; use --strict-format to check it")
    print(f"- answer extraction: {extractor_name}")
    print()

    for row in rows:
        row_id = row.get("id", row.get("_line_no", "?"))
        traces = row.get("generated_traces") or []
        expected_n = (row.get("sampling") or {}).get("n")

        if not traces:
            rows_without_traces += 1
            issue_count += 1
            if issue_count <= args.max_issues:
                status = row.get("status")
                error = row.get("error")
                print(f"[ROW {row_id}] no generated_traces (status={status!r}, error={error!r})")
            continue

        if expected_n is not None and len(traces) != expected_n:
            issue_count += 1
            if issue_count <= args.max_issues:
                print(f"[ROW {row_id}] expected {expected_n} traces from sampling.n, found {len(traces)}")

        for idx, trace in enumerate(traces):
            total_traces += 1
            result = check_trace(row, trace, idx, answer_extractor, args.strict_format)
            parseable_traces += int(result["ok"])
            should_print = args.show_ok or result["failures"] or result["warnings"]
            if should_print and issue_count < args.max_issues:
                label = "PARSEABLE" if result["ok"] else "BAD"
                if result["warnings"] and result["ok"]:
                    label = "WARN"
                print(f"[{label} row={row_id} trace={idx}] extracted={result['extracted']!r}")
                for failure in result["failures"]:
                    print(f"  FAIL: {failure}")
                for warning in result["warnings"]:
                    print(f"  WARN: {warning}")
                issue_count += int(bool(result["failures"] or result["warnings"]) or args.show_ok)

    bad_traces = total_traces - parseable_traces
    print()
    print("Summary:")
    print(f"- rows checked: {total_rows}")
    print(f"- rows without traces: {rows_without_traces}")
    print(f"- traces checked: {total_traces}")
    print(f"- parseable traces: {parseable_traces}")
    print(f"- unparseable traces: {bad_traces}")

    return 1 if rows_without_traces or bad_traces else 0


if __name__ == "__main__":
    raise SystemExit(main())
