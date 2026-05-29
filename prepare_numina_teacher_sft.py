"""Prepare a NuminaMath-CoT slice for teacher-model SFT.

The output mirrors the inference-time prompt path in ``prompt_strategy.py``:
each record is a chat transcript with the same system prompt and user prompt
that ``quick_eval.py`` would send to the model, followed by a boxed assistant
solution from NuminaMath-CoT.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from prompt_strategy import build_prompt, classify_math_topic


DATASET_ID = "AI-MO/NuminaMath-CoT"
BOXED_MARKER = "\\boxed{"
DEFAULT_SHARDS = tuple(f"data/train-{index:05d}-of-00005.parquet" for index in range(5))


class Progress:
    """Tiny progress wrapper with tqdm support when it is installed."""

    def __init__(self, total: int, *, enabled: bool, every: int) -> None:
        self.total = total
        self.enabled = enabled
        self.every = max(every, 1)
        self.accepted = 0
        self.scanned = 0
        self.skipped = 0
        self._bar = None
        if enabled:
            try:
                from tqdm.auto import tqdm

                if sys.stderr.isatty():
                    self._bar = tqdm(total=total, desc="Preparing Numina", unit="rows", mininterval=0.5)
                else:
                    print("Preparing Numina rows...", file=sys.stderr)
            except ImportError:
                print("Preparing Numina rows...", file=sys.stderr)

    def update(self, *, scanned: int, accepted: int, skipped: int) -> None:
        if not self.enabled:
            return

        delta = accepted - self.accepted
        self.scanned = scanned
        self.accepted = accepted
        self.skipped = skipped

        if self._bar is not None:
            if delta:
                self._bar.update(delta)
            self._bar.set_postfix(scanned=scanned, skipped=skipped)
        elif accepted == self.total or scanned % self.every == 0:
            print(
                f"accepted={accepted}/{self.total} scanned={scanned} skipped={skipped}",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        elif self.enabled:
            print(file=sys.stderr)


def finish(args: argparse.Namespace, code: int = 0) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(code)


def has_boxed_answer(text: str) -> bool:
    """Return True when text contains a balanced ``\\boxed{...}`` expression."""

    start = text.find(BOXED_MARKER)
    while start != -1:
        index = start + len(BOXED_MARKER)
        depth = 1
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return True
            index += 1
        start = text.find(BOXED_MARKER, start + len(BOXED_MARKER))
    return False


def clean_answer_candidate(candidate: str) -> str:
    """Trim common prose and display-math wrappers around a final answer."""

    candidate = candidate.strip()
    candidate = re.sub(r"^\$+\s*|\s*\$+$", "", candidate)
    candidate = re.sub(r"^\\\[\s*|\s*\\\]$", "", candidate)
    candidate = re.sub(r"^\\\(\s*|\s*\\\)$", "", candidate)
    candidate = candidate.strip(" .,:;")
    candidate = re.sub(r"^(therefore|thus|hence|so),?\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"^(the\s+)?(final\s+)?answer\s+(is|=|:)\s+", "", candidate, flags=re.I)
    return candidate.strip(" .,:;")


def extract_final_answer_candidate(solution: str) -> str | None:
    """Best-effort extraction for unboxed Numina solutions.

    The script only uses this for rows with an explicit final-answer cue. This
    keeps the generated SFT data cleaner than blindly boxing the last sentence.
    """

    tail = "\n".join(line.strip() for line in solution.splitlines()[-8:] if line.strip())
    patterns = (
        r"(?:final answer|answer|result|value|solution)\s*(?:is|=|:)\s*(.+?)(?:$|\n)",
        r"(?:therefore|thus|hence|so),?\s+(.+?)(?:$|\n)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, tail, flags=re.I | re.S)
        for match in reversed(matches):
            candidate = clean_answer_candidate(match)
            if candidate and len(candidate) <= 240 and BOXED_MARKER not in candidate:
                return candidate
    return None


def ensure_boxed_solution(solution: str, *, box_mode: str) -> str | None:
    """Return a solution containing a boxed answer, or None if it is ambiguous."""

    solution = solution.strip()
    if has_boxed_answer(solution):
        return solution

    candidate = extract_final_answer_candidate(solution)
    if candidate:
        return f"{solution}\n\nThus, the final answer is \\boxed{{{candidate}}}."

    if box_mode == "wrap-last-line":
        last_line = next((line.strip() for line in reversed(solution.splitlines()) if line.strip()), "")
        candidate = clean_answer_candidate(last_line)
        if candidate:
            return f"{solution}\n\nThus, the final answer is \\boxed{{{candidate}}}."

    return None


def iter_numina_rows_from_parquet(args: argparse.Namespace) -> Iterable[dict]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install `huggingface_hub` and `pyarrow`, "
            "or run with `--source-mode datasets-streaming`."
        ) from exc

    for shard in args.shards:
        shard_path = hf_hub_download(
            repo_id=args.dataset,
            filename=shard,
            repo_type="dataset",
        )
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            yield from batch.to_pylist()


def iter_numina_rows_from_datasets(args: argparse.Namespace) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install Hugging Face datasets first, e.g. "
            "`pip install datasets`."
        ) from exc

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=False,
    )
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return iter(dataset)


def iter_numina_rows(args: argparse.Namespace) -> Iterable[dict]:
    if args.source_mode == "parquet":
        return iter_numina_rows_from_parquet(args)
    return iter_numina_rows_from_datasets(args)


def build_record(row: dict, row_index: int, args: argparse.Namespace) -> dict | None:
    problem = row.get("problem") or row.get("messages", [{}])[0].get("content")
    solution = row.get("solution") or row.get("messages", [{}, {}])[1].get("content")
    if not problem or not solution:
        return None

    boxed_solution = ensure_boxed_solution(solution, box_mode=args.box_mode)
    if boxed_solution is None:
        return None

    topic = classify_math_topic(problem)
    system_prompt, user_prompt = build_prompt(problem, topic=topic)
    return {
        "id": f"numina_{row_index}",
        "source": row.get("source"),
        "topic": topic.name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": boxed_solution},
        ],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 5k NuminaMath-CoT teacher-SFT JSONL with inference prompts."
    )
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/numina_teacher_sft_5k.jsonl"))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--source-mode",
        choices=("parquet", "datasets-streaming"),
        default="parquet",
        help="parquet downloads local shards and exits cleanly; datasets-streaming uses load_dataset streaming.",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        default=list(DEFAULT_SHARDS),
        help="Parquet shard paths inside the Hugging Face dataset repo.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=100000,
        help="Maximum source rows to scan while looking for boxed/boxable examples.",
    )
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Streaming shuffle buffer. Leave at 0 to take the deterministic dataset order.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="For --source-mode datasets-streaming, load the split normally instead of streaming.",
    )
    parser.set_defaults(streaming=True)
    parser.add_argument(
        "--box-mode",
        choices=("strict", "wrap-last-line"),
        default="strict",
        help=(
            "strict keeps existing boxed answers and obvious final-answer phrases; "
            "wrap-last-line also boxes the last non-empty line when needed."
        ),
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print this many prepared records instead of writing the full file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable the progress bar/status output.",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Fallback status interval when tqdm is not installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    scanned = 0
    skipped = 0
    progress = Progress(args.limit, enabled=args.progress and not args.preview, every=args.progress_every)

    try:
        for row_index, row in enumerate(iter_numina_rows(args)):
            if scanned >= args.max_scan or len(records) >= args.limit:
                break
            scanned += 1
            record = build_record(row, row_index, args)
            if record is None:
                skipped += 1
            else:
                records.append(record)
            progress.update(scanned=scanned, accepted=len(records), skipped=skipped)
    finally:
        progress.close()

    if len(records) < args.limit:
        raise SystemExit(
            f"Only prepared {len(records)} records after scanning {scanned}. "
            "Increase --max-scan or use --box-mode wrap-last-line."
        )

    if args.preview:
        for record in records[: args.preview]:
            print(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"Previewed {min(args.preview, len(records))} of {len(records)} prepared records.")
        finish(args)

    write_jsonl(args.output, records)
    print(f"Scanned {scanned} rows; skipped {skipped}; wrote {len(records)} records to {args.output}")
    finish(args)


if __name__ == "__main__":
    main()
