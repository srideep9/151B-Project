from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal


APP_NAME = "child-public-clean-finetune"

# Child model from run_task.py.
BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# Train only on the cleaned public set.
TRAIN_FILE = "data/public_run_1_clean.jsonl"

REMOTE_OUTPUT_DIR = "/checkpoints/qwen3-4b-public-clean-lora"

# Keep this small and reproducible for the final handoff.
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_EPOCHS = 2.0
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
SEED = 151

# QLoRA lets this fit on a single large GPU. H100 is a good speed/cost balance.
GPU_TYPE = "H100"


CHECKPOINT_VOL = modal.Volume.from_name("child-finetune-checkpoints", create_if_missing=True)
CACHE_VOL = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch",
            "transformers==4.56.2",
            "tokenizers==0.22.1",
            "accelerate",
            "datasets",
            "peft",
            "bitsandbytes",
            "sentencepiece",
            "huggingface_hub",
        ]
    )
)

app = modal.App(APP_NAME)


SYSTEM_PROMPT_MATH = (
    "You are an MIT mathematician.\n"
    "Solve the problem using extremely concise internal reasoning inside <think> tags. "
    "Keep reasoning minimal, precise, and computation-focused. "
    "Avoid any conversational text, repetition, and unnecessary verification. No explanations or narrative sentences.\n"
    "Then output the final answer(s) inside a single \\boxed{}.\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Do not include units or labels inside \\boxed{}.\n"
    "- If the problem has multiple sub-answers or multiple [ANS] blanks, output the answers in the exact order they are requested, separated by commas, inside one box, e.g., \\boxed{3, 7, yes}.\n"
    "- Preserve required parentheses, brackets, and interval notation, e.g., \\boxed{(2, -2)}.\n"
    "- Always prefer exact symbolic forms for answers, no matter the type of question being asked. If a decimal is required, you MUST provide the answer to at least 6 decimal place.\n"
    "- NEVER debate or second-guess formatting expectations inside the <think> tags. Once you have derived the required values, immediately output the \\boxed{} and stop.\n"
    "- Do not output anything after the boxed answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an MIT mathematician.\n"
    "Solve the problem using extremely concise internal reasoning inside <think> tags. "
    "Keep reasoning minimal, precise, and computation-focused. "
    "Avoid any conversational text, repetition, and unnecessary verification. No explanations or narrative sentences.\n"
    "Then output ONLY the final multiple-choice answer as a single uppercase letter inside a \\boxed{}.\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Output exactly one boxed uppercase letter as your answer, e.g., \\boxed{C}\n"
    "- Do not output the answer text or numeric value.\n"
    "- Do not include punctuation inside the box.\n"
    "- NEVER debate or second-guess formatting expectations inside the <think> tags. Once you derive the answer, immediately output the \\boxed{} letter and stop.\n"
    "- Do not output anything after the boxed answer."
)


def build_messages(question: str, options: Optional[list], assistant_trace: Optional[str] = None) -> List[Dict[str, str]]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options))
        user_text = f"{question}\n\nOptions:\n{opts_text}"
        system_prompt = SYSTEM_PROMPT_MCQ
    else:
        user_text = question
        system_prompt = SYSTEM_PROMPT_MATH

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    if assistant_trace is not None:
        messages.append({"role": "assistant", "content": assistant_trace.strip()})
    return messages


def pick_training_trace(record: Dict[str, Any]) -> str:
    selected_trace = record.get("selected_trace")
    if isinstance(selected_trace, str) and selected_trace.strip():
        return selected_trace

    generated_traces = record.get("generated_traces") or []
    for trace in generated_traces:
        if isinstance(trace, str) and trace.strip():
            return trace

    raise ValueError(f"Record {record.get('id')} has no usable trace.")


def load_training_records(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("answer_match") is not True or record.get("status") != "correct":
                continue
            if not record.get("question"):
                continue
            rows.append(record)
    return rows


def make_sft_dataset(records: List[Dict[str, Any]], tokenizer) -> Any:
    from datasets import Dataset

    examples = []
    for record in records:
        trace = pick_training_trace(record)
        prompt_messages = build_messages(
            question=record["question"],
            options=record.get("options"),
            assistant_trace=None,
        )
        full_messages = build_messages(
            question=record["question"],
            options=record.get("options"),
            assistant_trace=trace,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).input_ids
        labels = list(full_ids)
        prompt_label_len = min(len(prompt_ids), len(labels))
        labels[:prompt_label_len] = [-100] * prompt_label_len
        if all(label == -100 for label in labels):
            continue
        examples.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
                "id": record.get("id"),
            }
        )

    print(f"Loaded {len(examples)} clean SFT examples")
    return Dataset.from_list(examples)


def make_data_collator(tokenizer):
    def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])
            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("hf_token")],
    volumes={
        "/checkpoints": CHECKPOINT_VOL,
        "/cache": CACHE_VOL,
    },
)
def train_child_lora(
    train_records: List[Dict[str, Any]],
    hf_repo_id: Optional[str] = None,
    private_hf_repo: bool = False,
    base_model_id: str = BASE_MODEL_ID,
    output_dir: str = REMOTE_OUTPUT_DIR,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface/transformers"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if hf_repo_id and not hf_token:
        raise RuntimeError(
            "hf_repo_id was provided, but HF_TOKEN is not available inside the Modal container. "
            "Create/update the Modal secret with: modal secret create hf_token HF_TOKEN=..."
        )

    print("base_model_id:", base_model_id)
    print("num_train_records:", len(train_records))
    print("output_dir:", output_dir)
    print("hf_repo_id:", hf_repo_id)
    print("private_hf_repo:", private_hf_repo)
    print("hf_token_available:", bool(hf_token))
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir="/cache/huggingface/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir="/cache/huggingface/hub",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    dataset = make_sft_dataset(train_records, tokenizer)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=make_data_collator(tokenizer),
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    CHECKPOINT_VOL.commit()
    CACHE_VOL.commit()

    if hf_repo_id:
        from huggingface_hub import HfApi

        print(f"Pushing LoRA adapter to Hugging Face Hub: {hf_repo_id}")
        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=hf_repo_id,
            repo_type="model",
            private=private_hf_repo,
            exist_ok=True,
        )
        trainer.model.push_to_hub(hf_repo_id, private=private_hf_repo, token=hf_token)
        tokenizer.push_to_hub(hf_repo_id, private=private_hf_repo, token=hf_token)

    return {
        "base_model_id": base_model_id,
        "output_dir": output_dir,
        "hf_repo_id": hf_repo_id,
        "private_hf_repo": private_hf_repo,
        "num_examples": len(dataset),
        "adapter_type": "lora",
    }


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the child model on clean public traces using Modal.")
    parser.add_argument(
        "--hf_repo_id",
        default=None,
        help="Optional Hugging Face repo id to push the LoRA adapter, e.g. username/qwen3-4b-public-clean-lora.",
    )
    parser.add_argument("--private_hf_repo", action="store_true", help="Create/upload the HF adapter repo as private.")
    parser.add_argument("--base_model_id", default=BASE_MODEL_ID)
    parser.add_argument("--train_file", default=TRAIN_FILE)
    parser.add_argument("--output_dir", default=REMOTE_OUTPUT_DIR)
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    train_records = load_training_records(args.train_file)
    print(f"Loaded {len(train_records)} local clean training records from {args.train_file}")
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
