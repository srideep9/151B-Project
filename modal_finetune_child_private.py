from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from modal_finetune_child_public import BASE_MODEL_ID, app, train_child_lora


TRAIN_FILE = "private_teacher_traces_qwen25_math72b_modal_clean.jsonl"
REMOTE_OUTPUT_DIR = "/checkpoints/qwen3-4b-private-clean-lora"


def nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and any(
        (nonempty_str(item) if isinstance(item, str) else bool(item))
        for item in value
    )


def load_private_training_records(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "majority":
                continue
            if record.get("error"):
                continue
            if not record.get("question"):
                continue
            if not nonempty_str(record.get("selected_trace")):
                continue
            if not nonempty_str(record.get("majority_answer")):
                continue
            if not nonempty_list(record.get("generated_traces")):
                continue
            record["training_source"] = "private_majority"
            rows.append(record)
    return rows


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the child model on clean private majority traces using Modal.")
    parser.add_argument(
        "--hf_repo_id",
        default=None,
        help="Optional Hugging Face repo id to push the LoRA adapter, e.g. username/qwen3-4b-private-clean-lora.",
    )
    parser.add_argument("--private_hf_repo", action="store_true", help="Create/upload the HF adapter repo as private.")
    parser.add_argument("--base_model_id", default=BASE_MODEL_ID)
    parser.add_argument("--train_file", default=TRAIN_FILE)
    parser.add_argument("--output_dir", default=REMOTE_OUTPUT_DIR)
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    train_records = load_private_training_records(args.train_file)
    print(f"Loaded {len(train_records)} local clean private training records from {args.train_file}")
    with app.run():
        result = train_child_lora.remote(
            train_records=train_records,
            hf_repo_id=args.hf_repo_id,
            private_hf_repo=args.private_hf_repo,
            base_model_id=args.base_model_id,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
