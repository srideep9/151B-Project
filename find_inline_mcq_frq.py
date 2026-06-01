import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


OPTION_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-Z])\.\s+")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def answer_is_letters(answer: Any) -> bool:
    parts = answer if isinstance(answer, list) else [answer]
    return bool(parts) and all(
        isinstance(part, str) and re.fullmatch(r"[A-Z]+", part.strip())
        for part in parts
    )


def find_option_runs(question: str) -> List[Dict[str, Any]]:
    matches = list(OPTION_LABEL_RE.finditer(question))
    runs = []
    cur = []

    for match in matches:
        label = match.group(1)
        if not cur:
            cur = [match]
            continue

        prev_label = cur[-1].group(1)
        expected_next = chr(ord(prev_label) + 1)
        if label == expected_next:
            cur.append(match)
        else:
            if len(cur) >= 2:
                runs.append(run_to_record(question, cur))
            cur = [match]

    if len(cur) >= 2:
        runs.append(run_to_record(question, cur))

    return runs


def run_to_record(question: str, matches: List[re.Match]) -> Dict[str, Any]:
    labels = [m.group(1) for m in matches]
    start = matches[0].start()
    end = matches[-1].end()
    snippet_start = max(0, start - 180)
    snippet_end = min(len(question), matches[-1].start() + 500)
    return {
        "labels": labels,
        "num_options": len(labels),
        "char_start": start,
        "char_end": end,
        "snippet": question[snippet_start:snippet_end],
    }


def likely_inline_mcq(row: Dict[str, Any]) -> bool:
    if row.get("options"):
        return False
    question = row.get("question") or ""
    if "[ANS]" not in question:
        return False
    runs = find_option_runs(question)
    if not runs:
        return False
    return answer_is_letters(row.get("answer")) or len(runs) >= question.count("[ANS]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/public.jsonl")
    parser.add_argument("--output", default="data/public_inline_mcq_frq.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_jsonl(input_path)

    matches = []
    no_options = 0
    for row in rows:
        if not row.get("options"):
            no_options += 1
        if not likely_inline_mcq(row):
            continue
        question = row.get("question") or ""
        option_runs = find_option_runs(question)
        matches.append({
            "id": row.get("id"),
            "line_no": row.get("_line_no"),
            "answer": row.get("answer"),
            "num_ans_blanks": question.count("[ANS]"),
            "num_option_runs": len(option_runs),
            "option_runs": option_runs,
            "question": question,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in matches:
            f.write(json.dumps(row) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Rows with no options field: {no_options}")
    print(f"Inline-MCQ-looking FRQ rows: {len(matches)}")
    print(f"Wrote: {output_path}")
    print("First few ids:", [row["id"] for row in matches[:20]])


if __name__ == "__main__":
    main()
