import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from judger import Judger


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({
                    "_line_no": line_no,
                    "_json_error": str(exc),
                    "_raw_line": line[:1000],
                })
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def extract_boxed_contents(text: str) -> List[str]:
    contents = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            contents.append(text[brace_start:i - 1].strip())
        else:
            contents.append("__MALFORMED_BOX__")
            break
        start = i
    return contents


def response_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []
    if row.get("selected_trace"):
        candidates.append({"source": "selected_trace", "index": None, "text": row["selected_trace"]})
    elif row.get("response"):
        candidates.append({"source": "response", "index": None, "text": row["response"]})

    for idx, trace in enumerate(row.get("generated_traces") or []):
        candidates.append({"source": "generated_traces", "index": idx, "text": trace})

    if not candidates:
        candidates.append({"source": "none", "index": None, "text": ""})
    return candidates


def expected_parts(row: Dict[str, Any]) -> Optional[int]:
    expected = row.get("expected_answer", row.get("gold", row.get("answer")))
    if isinstance(expected, list):
        return len(expected)
    if expected is None:
        return None
    return 1


def is_probably_mcq(row: Dict[str, Any]) -> bool:
    expected = row.get("expected_answer", row.get("gold", row.get("answer")))
    if row.get("options"):
        return True
    if isinstance(expected, str) and re.fullmatch(r"[A-Z]+", expected.strip()):
        return True
    if isinstance(expected, list) and all(isinstance(x, str) and re.fullmatch(r"[A-Z]", x.strip()) for x in expected):
        return True
    return False


def tail(text: str, n: int = 900) -> str:
    text = text.replace("\r", "")
    return text[-n:] if len(text) > n else text


def audit_row(row: Dict[str, Any], judger: Judger) -> Optional[Dict[str, Any]]:
    if row.get("_json_error"):
        return {
            "id": row.get("id"),
            "line_no": row["_line_no"],
            "risk_flags": ["json_decode_error"],
            "json_error": row["_json_error"],
            "raw_line": row["_raw_line"],
        }

    row_flags = []
    if row.get("status") not in (None, "correct", "majority"):
        row_flags.append(f"status:{row.get('status')}")
    if row.get("answer_match") is False:
        row_flags.append("answer_match_false")
    if row.get("error"):
        row_flags.append("row_error")
    if row.get("vote_status") not in (None, "majority"):
        row_flags.append(f"vote_status:{row.get('vote_status')}")
    if any(reason == "length" for reason in row.get("finish_reasons") or []):
        row_flags.append("finish_reason_length")

    exp_parts = expected_parts(row)
    mcq = is_probably_mcq(row)
    candidate_reports = []

    for cand in response_candidates(row):
        text = cand["text"] or ""
        flags = []
        boxes = extract_boxed_contents(text)
        extracted = judger.extract_ans(text)
        split_extracted = judger.split_by_comma(extracted) if extracted else []

        if not text:
            flags.append("empty_response")
        if "\\boxed{" not in text:
            flags.append("no_boxed")
        if "__MALFORMED_BOX__" in boxes:
            flags.append("malformed_box")
        if len(boxes) > 3:
            flags.append("many_boxed_answers")
        if not extracted:
            flags.append("extract_ans_empty")
        if extracted and len(extracted) > 120:
            flags.append("very_long_extracted_answer")
        if extracted and re.search(r"\b(answer|option|therefore|hence|final)\b", extracted, re.I):
            flags.append("prose_inside_extracted_answer")
        if mcq and extracted and not re.fullmatch(r"[A-Z]+", extracted.strip()):
            flags.append("mcq_extracted_not_letters")
        if exp_parts is not None and extracted and not mcq and len(split_extracted) != exp_parts:
            flags.append(f"part_count_mismatch:{len(split_extracted)}!={exp_parts}")
        if "<think>" in text and "</think>" not in text:
            flags.append("unclosed_think")
        if text.count("\\boxed{") >= 2 and len(set(boxes)) > 1:
            flags.append("multiple_distinct_boxes")

        if flags:
            candidate_reports.append({
                "source": cand["source"],
                "index": cand["index"],
                "risk_flags": flags,
                "extracted": extracted,
                "split_extracted": split_extracted,
                "boxed_count": len(boxes),
                "boxed_unique": sorted(set(boxes))[:12],
                "finish_reason": (
                    row.get("finish_reasons", [None] * len(row.get("generated_traces") or []))[cand["index"]]
                    if cand["source"] == "generated_traces"
                    and cand["index"] is not None
                    and cand["index"] < len(row.get("finish_reasons") or [])
                    else None
                ),
                "response_tail": tail(text),
            })

    if not row_flags and not candidate_reports:
        return None

    expected = row.get("expected_answer", row.get("gold", row.get("answer")))
    return {
        "id": row.get("id"),
        "line_no": row.get("_line_no"),
        "question": row.get("question"),
        "expected_answer": expected,
        "predicted_answer": row.get("predicted_answer"),
        "answer_match": row.get("answer_match"),
        "status": row.get("status"),
        "vote_status": row.get("vote_status"),
        "majority_answer": row.get("majority_answer"),
        "risk_flags": row_flags,
        "candidate_reports": candidate_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/public_run_1.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_bad_generations.jsonl")

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[:args.limit]

    judger = Judger(strict_extract=False)
    bad_rows = []
    flag_counts = Counter()
    candidate_flag_counts = Counter()

    for row in rows:
        report = audit_row(row, judger)
        if report is None:
            continue
        bad_rows.append(report)
        flag_counts.update(report.get("risk_flags") or [])
        for cand in report.get("candidate_reports") or []:
            candidate_flag_counts.update(cand.get("risk_flags") or [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for report in bad_rows:
            f.write(json.dumps(report) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Flagged rows: {len(bad_rows)}")
    print(f"Wrote: {output_path}")
    print("Row-level flags:")
    for flag, count in flag_counts.most_common():
        print(f"  {flag}: {count}")
    print("Candidate-level flags:")
    for flag, count in candidate_flag_counts.most_common():
        print(f"  {flag}: {count}")


if __name__ == "__main__":
    main()
