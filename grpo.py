 
from __future__ import annotations
 
import argparse
import json
import os
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional
 
import modal
 
# ─────────────────────────────────────────
# MODAL CONFIG
# ─────────────────────────────────────────
APP_NAME = "child-grpo-finetune"
GPU_TYPE = "H100"
 
CHECKPOINT_VOL = modal.Volume.from_name("child-finetune-checkpoints", create_if_missing=True)
CACHE_VOL      = modal.Volume.from_name("qwen-model-cache",           create_if_missing=True)
 
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "transformers==4.56.2",
        "tokenizers==0.22.1",
        "accelerate",
        "datasets",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "huggingface_hub",
        "trl>=0.12.0",   # needs GRPOTrainer
        "sympy",
        "numpy",
    ])
)
 
app = modal.App(APP_NAME)
 
# ─────────────────────────────────────────
# USER-CONFIGURABLE CONSTANTS
# ─────────────────────────────────────────
BASE_MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
REMOTE_OUTPUT_DIR = "/checkpoints/qwen3-4b-grpo-lora"
 
MAX_SEQ_LENGTH          = 4096
GRPO_MAX_NEW_TOKENS     = 2048
NUM_TRAIN_EPOCHS        = 1
LEARNING_RATE           = 5e-6
PER_DEVICE_TRAIN_BATCH  = 2      # questions per step
GRADIENT_ACCUMULATION   = 4      # effective batch = 8
NUM_GENERATIONS         = 6      # responses sampled per question
LORA_R                  = 32
LORA_ALPHA              = 64
LORA_DROPOUT            = 0.05
SEED                    = 151
 
# ─────────────────────────────────────────
# SYSTEM PROMPTS 
# ─────────────────────────────────────────
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
 
 
# ─────────────────────────────────────────
# DATA HELPERS (run locally to load records)
# ─────────────────────────────────────────
 
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
 
 
def build_prompt(question: str, options: Optional[list]) -> str:
    """Build the chat-formatted prompt string for a question."""
    system = SYSTEM_PROMPT_MCQ if options else SYSTEM_PROMPT_MATH
 
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
        )
        user_msg = f"{question}\n\nOptions:\n{opts_text}"
    else:
        user_msg = question
 
    # Use raw string format matching Qwen3 chat template
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
 
 
def load_grpo_records(path: str) -> List[Dict[str, Any]]:
    """
    Load questions for GRPO.
    """
    records = load_jsonl(path)
    examples = []
 
    for rec in records:
        question = rec.get("question", "")
        options  = rec.get("options")
        answer   = rec.get("answer")
 
        # Skip records without a question or ground truth
        if not question or answer is None:
            continue
 
        examples.append({
            "prompt":  build_prompt(question, options),
            "answer":  answer,
            "options": options,
        })
 
    print(f"Loaded {len(examples)} GRPO questions from {path}")
    return examples
 

# REWARD FUNCTIONS

 
class TimeoutException(Exception):
    pass
 
def _timeout_handler(signum, frame):
    raise TimeoutException()
 
 
def extract_boxed(text: str) -> Optional[str]:
    """Extract last \\boxed{} content after </think>."""
    idx    = text.rfind("</think>")
    search = text[idx + len("</think>"):] if idx >= 0 else text
 
    matches = []
    start   = 0
    while True:
        i = search.find("\\boxed{", start)
        if i < 0:
            break
        depth = 1
        j     = i + len("\\boxed{")
        while j < len(search) and depth > 0:
            if search[j] == "{":
                depth += 1
            elif search[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(search[i + len("\\boxed{"):j - 1].strip())
        start = j
 
    return matches[-1] if matches else None
 
 
def format_reward_fn(completions: List[str], options_list: List[Optional[list]]) -> List[float]:
    """
    +0.5  correct format  (boxed + right shape)
    -0.5  missing or wrong format
    """
    rewards = []
    for text, options in zip(completions, options_list):
        ans = extract_boxed(text)
        if ans is None:
            rewards.append(-0.5)
            continue
        if options:
            # MCQ: single uppercase letter
            rewards.append(0.5 if re.fullmatch(r"[A-Z]", ans.strip()) else -0.5)
        else:
            rewards.append(0.5 if ans.strip() else -0.5)
    return rewards
 
 
def correctness_reward_fn(
    completions: List[str],
    answers: List[Any],
    options_list: List[Optional[list]],
) -> List[float]:
    """
    +1.0  correct answer
    -1.0  wrong answer
     0.0  can't parse
    """
    # Import here — this runs inside the Modal container where judger is available
    from judger import Judger
    jud = Judger(strict_extract=False)
 
    rewards = []
    for text, gold, options in zip(completions, answers, options_list):
        predicted = extract_boxed(text)
        if predicted is None:
            rewards.append(0.0)
            continue
 
        gold_str = (
            ", ".join(str(g) for g in gold)
            if isinstance(gold, list)
            else str(gold)
        )
 
        try:
            if options:
                match = predicted.strip().upper() == gold_str.strip().upper()
            else:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)
                try:
                    list1 = jud.split_by_comma(predicted)
                    list2 = jud.split_by_comma(gold_str)
                    if len(list1) != len(list2):
                        match = False
                    else:
                        match = all(
                            jud.is_equal(
                                jud.norm_ans_str(a),
                                jud.norm_ans_str(b),
                            )
                            for a, b in zip(list1, list2)
                        )
                finally:
                    signal.alarm(0)
 
            rewards.append(1.0 if match else -1.0)
 
        except Exception:
            signal.alarm(0)
            rewards.append(0.0)
 
    return rewards
 
 
def combined_reward(completions, prompts=None, **kwargs) -> List[float]:
    """
    Single reward function passed to GRPOTrainer.
    Total range: -1.5 (worst) to +1.5 (best).
 
    GRPOTrainer passes per-example dataset columns as kwargs.
    We rely on 'answer' and 'options' being in the dataset.
    """
    answers      = kwargs.get("answer",  [None] * len(completions))
    options_list = kwargs.get("options", [None] * len(completions))
 
    fmt  = format_reward_fn(completions, options_list)
    corr = correctness_reward_fn(completions, answers, options_list)
 
    total = [f + c for f, c in zip(fmt, corr)]
 
    # Log reward breakdown to stdout for monitoring
    avg_fmt  = sum(fmt)  / len(fmt)
    avg_corr = sum(corr) / len(corr)
    avg_tot  = sum(total) / len(total)
    print(f"[reward] format={avg_fmt:.3f}  correctness={avg_corr:.3f}  total={avg_tot:.3f}")
 
    return total
 
 

# MODAL TRAINING FUNCTION

 
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("hf_token")],
    volumes={
        "/checkpoints": CHECKPOINT_VOL,
        "/cache":       CACHE_VOL,
    },
    # Upload judger.py + utils.py into the container so reward fn can import them
    mounts=[
        modal.Mount.from_local_file("judger.py", remote_path="/root/judger.py"),
        modal.Mount.from_local_file("utils.py",  remote_path="/root/utils.py"),
    ],
)
def train_grpo(
    train_records: List[Dict[str, Any]],
    hf_repo_id: Optional[str] = None,
    private_hf_repo: bool = False,
    base_model_id: str = BASE_MODEL_ID,
    sft_adapter_id: Optional[str] = None,
    output_dir: str = REMOTE_OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Run GRPO fine-tuning inside the Modal H100 container.
 
    Args:
        train_records:   list of {prompt, answer, options} dicts
        hf_repo_id:      optional HuggingFace repo to push adapter to
        private_hf_repo: whether the HF repo should be private
        base_model_id:   base model to start from
        sft_adapter_id:  optional HF repo or local path of SFT LoRA adapter
                         to merge before GRPO (recommended)
        output_dir:      where to save inside the Modal volume
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import GRPOTrainer, GRPOConfig
 
    os.environ["HF_HOME"]                = "/cache/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"]  = "/cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"]     = "/cache/huggingface/transformers"
 
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
 
    print("base_model_id:     ", base_model_id)
    print("sft_adapter_id:    ", sft_adapter_id)
    print("num_train_records: ", len(train_records))
    print("output_dir:        ", output_dir)
    print("hf_repo_id:        ", hf_repo_id)
    print("cuda_available:    ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:               ", torch.cuda.get_device_name(0))
 
    # ── Tokenizer ────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir="/cache/huggingface/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # GRPO needs left-padding for generation
 
    # ── Model ────────────────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
 
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir="/cache/huggingface/hub",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
 
    # ── Load SFT adapter if provided ─────────────────────────────────────────────
    if sft_adapter_id:
        print(f"Loading SFT adapter from: {sft_adapter_id}")
        # Download if it's a HF repo id
        if not Path(sft_adapter_id).exists():
            from huggingface_hub import snapshot_download
            sft_adapter_id = snapshot_download(
                repo_id=sft_adapter_id,
                token=hf_token,
                cache_dir="/cache/huggingface/hub",
            )
        model = PeftModel.from_pretrained(model, sft_adapter_id, is_trainable=True)
        print("SFT adapter loaded successfully.")
    else:
        # Fresh LoRA on top of base model
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
 
    model.print_trainable_parameters()
 
    # ── Dataset ───────────────────────────────────────────────────────────────────
    dataset = Dataset.from_list(train_records)
    print(f"GRPO dataset size: {len(dataset)}")
 
    # ── GRPO Config ───────────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        # GRPO-specific
        num_generations=NUM_GENERATIONS,
        max_new_tokens=GRPO_MAX_NEW_TOKENS,
        max_prompt_length=MAX_SEQ_LENGTH - GRPO_MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
        # Keep memory usage reasonable
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )
 
    # ── Trainer ───────────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=combined_reward,
        tokenizer=tokenizer,
    )
 
    print("[GRPO] Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[GRPO] Model saved to {output_dir}")
 
    CHECKPOINT_VOL.commit()
    CACHE_VOL.commit()
 
    # ── Push to HuggingFace ───────────────────────────────────────────────────────
    if hf_repo_id:
        from huggingface_hub import HfApi
 
        print(f"Pushing GRPO adapter to HuggingFace: {hf_repo_id}")
        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=hf_repo_id,
            repo_type="model",
            private=private_hf_repo,
            exist_ok=True,
        )
        trainer.model.push_to_hub(hf_repo_id, private=private_hf_repo, token=hf_token)
        tokenizer.push_to_hub(hf_repo_id, private=private_hf_repo, token=hf_token)
        print(f"Adapter pushed to: https://huggingface.co/{hf_repo_id}")
 
    return {
        "base_model_id":  base_model_id,
        "sft_adapter_id": sft_adapter_id,
        "output_dir":     output_dir,
        "hf_repo_id":     hf_repo_id,
        "num_examples":   len(dataset),
        "adapter_type":   "grpo_lora",
    }
 
 
# ─────────────────────────────────────────
# LOCAL ENTRYPOINT
# ─────────────────────────────────────────
 
def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO fine-tune Qwen3-4B on math reasoning via Modal."
    )
    parser.add_argument(
        "--train_file",
        default="data/public.jsonl",
        help="Primary JSONL file with questions + ground truth answers.",
    )
    parser.add_argument(
        "--train_file_2",
        default=None,
        help="Optional second JSONL file (e.g. private.jsonl) to combine with train_file.",
    )
    parser.add_argument(
        "--sft_adapter_id",
        default="username/qwen3-4b-public-clean-lora",
        help=(
            "HuggingFace repo id or local path of the SFT LoRA adapter to start from. "
            "Set to 'none' to start GRPO from the base model directly."
        ),
    )
    parser.add_argument(
        "--hf_repo_id",
        default=None,
        help="HuggingFace repo to push the GRPO adapter to, e.g. username/qwen3-4b-grpo-lora.",
    )
    parser.add_argument(
        "--private_hf_repo",
        action="store_true",
        help="Make the HuggingFace repo private.",
    )
    parser.add_argument("--base_model_id", default=BASE_MODEL_ID)
    parser.add_argument("--output_dir",    default=REMOTE_OUTPUT_DIR)
    return parser
 
 
def main() -> None:
    args = create_arg_parser().parse_args()
 
    train_records = load_grpo_records(args.train_file)
 
    if args.train_file_2:
        extra = load_grpo_records(args.train_file_2)
        train_records = train_records + extra
        print(f"Combined dataset size: {len(train_records)}")
 
    sft_adapter = None if args.sft_adapter_id == "none" else args.sft_adapter_id
 
    print(f"Sending {len(train_records)} questions to Modal for GRPO training...")
    with app.run():
        result = train_grpo.remote(
            train_records=train_records,
            hf_repo_id=args.hf_repo_id,
            private_hf_repo=args.private_hf_repo,
            base_model_id=args.base_model_id,
            sft_adapter_id=sft_adapter,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
 
 
if __name__ == "__main__":
    main()