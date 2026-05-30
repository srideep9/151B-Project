from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from modal_finetune_child_private import load_private_training_records
from modal_finetune_child_public import BASE_MODEL_ID, app, load_training_records, train_child_lora


PUBLIC_TRAIN_FILE = "data/public_run_1_clean.jsonl"
PRIVATE_TRAIN_FILE = "private_teacher_traces_qwen25_math72b_modal_clean.jsonl"
REMOTE_OUTPUT_DIR = "/checkpoints/qwen3-4b-public-private-clean-lora"


def load_combined_training_records(public_path: str, private_path: str) -> List[Dict[str, Any]]:
    public_records = load_training_records(public_path)
    for record in public_records:
        record["training_source"] = "public_correct"

    private_records = load_private_training_records(private_path)
    return public_records + private_records


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the child model on clean public-correct and private-majority traces using Modal."
    )
    parser.add_argument(
        "--hf_repo_id",
        default=None,
        help="Optional Hugging Face repo id to push the LoRA adapter, e.g. username/qwen3-4b-public-private-clean-lora.",
    )
    parser.add_argument("--private_hf_repo", action="store_true", help="Create/upload the HF adapter repo as private.")
    parser.add_argument("--base_model_id", default=BASE_MODEL_ID)
    parser.add_argument("--public_train_file", default=PUBLIC_TRAIN_FILE)
    parser.add_argument("--private_train_file", default=PRIVATE_TRAIN_FILE)
    parser.add_argument("--output_dir", default=REMOTE_OUTPUT_DIR)
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    train_records = load_combined_training_records(args.public_train_file, args.private_train_file)
    source_counts: Dict[str, int] = {}
    for record in train_records:
        source = str(record.get("training_source"))
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Loaded {len(train_records)} local combined training records")
    print("source_counts:", json.dumps(source_counts, sort_keys=True))

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
