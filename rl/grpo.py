from __future__ import annotations

import json
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ─────────────────────────────────────────
# MODAL CONFIG
# ─────────────────────────────────────────
APP_NAME = "child-grpo-finetune"
GPU_TYPE = "H200:2"

CHECKPOINT_VOL = modal.Volume.from_name("child-finetune-checkpoints", create_if_missing=True)
CACHE_VOL      = modal.Volume.from_name("qwen-model-cache",           create_if_missing=True)

# Image includes vLLM for fast rollouts and removes bitsandbytes quantization
image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel")
    .env({
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True"
    })
    .pip_install([
        "torch",
        "transformers", 
        "tokenizers",
        "accelerate",
        "datasets",
        "peft",
        "vllm",
        "sentencepiece",
        "huggingface_hub",
        "trl",
        "sympy",
        "tqdm",
        "numpy",
        "antlr4-python3-runtime==4.11.1",
        "hf_transfer"
    ])
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("judger.py", remote_path="/root/judger.py")
)

app = modal.App(APP_NAME)

# ─────────────────────────────────────────
# USER-CONFIGURABLE CONSTANTS
# ─────────────────────────────────────────
BASE_MODEL_ID    = "hzia360/qwen3-4b-sft-merged2" 
REMOTE_OUTPUT_DIR = "/checkpoints/qwen3-4b-grpo-lora"

MAX_SEQ_LENGTH          = 8192
GRPO_MAX_NEW_TOKENS     = 4096
NUM_TRAIN_EPOCHS        = 1
LEARNING_RATE           = 5e-6
PER_DEVICE_TRAIN_BATCH  = 8      # questions per step
GRADIENT_ACCUMULATION   = 4      # effective batch = 8
NUM_GENERATIONS         = 4      # responses sampled per question
LORA_R                  = 32
LORA_ALPHA              = 64
LORA_DROPOUT            = 0.0
SEED                    = 167

# ─────────────────────────────────────────
# SYSTEM PROMPTS 
# ─────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an MIT mathematician.\n"
    "Solve the problem by thinking deeply inside <think> tags. "
    "You must think step-by-step, write out your detailed mathematical derivations, and explain your internal logic. "
    "Explore multiple paths, verify your work carefully, and correct yourself if you make a mistake. Take as much space as needed to guarantee the correct mathematical result.\n"
    "Then output the final answer(s) inside a single \\boxed{}.\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Do not include units or labels inside \\boxed{}.\n"
    "- If the problem has multiple [ANS] blanks, output the answers in the exact order they are requested, separated by commas, inside one box. The number of comma-separated items inside your \\boxed{} MUST exactly match the number of [ANS] placeholders in the question.\n"
    "- If there is only ONE [ANS] placeholder, but the solution has multiple values, you MUST group them inside parentheses seperated by commas. Example: \\boxed{(7, -7)}.\n"
    "- Always prefer exact symbolic forms for answers. Do not convert fractions to decimals. If a decimal is required, you MUST provide the answer to at least 6 decimal places. Never round or truncate intermediate values. Carry full precision through every step.\n"
    "- NEVER debate or second-guess formatting expectations inside the <think> tags. Once derived, immediately output the \\boxed{} and stop.\n"
    "- Do not output anything after the boxed answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an MIT mathematician.\n"
    "Solve the problem by thinking deeply inside <think> tags. "
    "You must think step-by-step, write out your detailed mathematical derivations, and explain your internal logic. "
    "Explore multiple paths, verify your work carefully, and correct yourself if you make a mistake. Take as much space as needed to guarantee the correct mathematical result.\n"
    "Then output ONLY the final multiple-choice answer as a single uppercase letter inside a \\boxed{}.\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Output exactly one boxed uppercase letter as your answer, e.g., \\boxed{C}\n"
    "- Do not output the answer text or numeric value.\n"
    "- Do not include punctuation inside the box.\n"
    "- NEVER debate or second-guess formatting expectations inside the <think> tags. Once derived, immediately output the \\boxed{} and stop.\n"
    "- Do not output anything after the boxed answer."
)


# ─────────────────────────────────────────
# DATA HELPERS 
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
    system = SYSTEM_PROMPT_MCQ if options else SYSTEM_PROMPT_MATH

    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
        )
        user_msg = f"{question}\n\nOptions:\n{opts_text}"
    else:
        user_msg = question

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_grpo_records(path: str) -> List[Dict[str, Any]]:
    records = load_jsonl(path)
    examples = []

    for rec in records:
        question = rec.get("question", "")
        options  = rec.get("options")
        answer   = rec.get("answer")

        # Skip records without a question or ground truth
        if not question or answer is None:
            continue

        # ── PYARROW TYPE NORMALIZATION ──
        
        # 1. Force answer to ALWAYS be a list of strings
        if not isinstance(answer, list):
            norm_answer = [str(answer)]
        else:
            norm_answer = [str(a) for a in answer]
            
        # 2. Force options to ALWAYS be a list of strings (empty list instead of None)
        if not options:
            norm_options = []
        else:
            norm_options = [str(opt) for opt in options]

        examples.append({
            # Note: We still pass the original 'options' to build_prompt so it 
            # correctly triggers the MCQ vs FRQ system prompt.
            "prompt":  build_prompt(question, options), 
            "answer":  norm_answer,
            "options": norm_options,
        })

    print(f"Loaded {len(examples)} GRPO questions from {path}")
    return examples


# ─────────────────────────────────────────
# REWARD FUNCTIONS
# ─────────────────────────────────────────

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()


def extract_boxed(text: str) -> Optional[str]:
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


def format_reward_fn(completions: List[str], options_list: List[Optional[list]], **kwargs) -> List[float]:
    """
    Evaluates formatting and penalizes lazy reasoning (lucky guesses).
    Returns +0.5 for perfect format, -0.5 for broken format, or -1.0 for skipping the <think> step.
    """
    rewards = []
    for text, options in zip(completions, options_list):
        ans = extract_boxed(text)
        
        # Fast extraction of <think> tags
        think_start = text.find("<think>")
        think_end = text.rfind("</think>")
        
        think_length = 0
        if think_start >= 0 and think_end > think_start:
            think_length = len(text[think_start + 7 : think_end].strip())
        
        # Penalize missing \boxed{} completely
        if ans is None:
            rewards.append(-0.5)
            continue
            
        # The Laziness Penalty: -1.0 if they didn't generate enough scratchpad reasoning
        if think_length < 100:
            rewards.append(-1.0) 
            continue

        # Check for strict MCQ format (single uppercase letter) vs FRQ format (any text)
        if options:
            rewards.append(0.3 if re.fullmatch(r"[A-Z]", ans.strip()) else -0.5)
        else:
            rewards.append(0.3 if len(ans.strip()) > 0 else -0.5)
            
    return rewards


def correctness_reward_fn(
    completions: List[str],
    answers: List[Any],
    options_list: List[Optional[list]],
    **kwargs
) -> List[float]:
    """
    Evaluates mathematical correctness exactly matching the inference grading script.
    Returns +1.0 for a match, -1.0 for a wrong answer.
    """
    from judger import Judger
    jud = Judger(strict_extract=False)

    # Helper function exactly matching inference logic
    def extract_letter(text: str) -> str:
        m = re.search(r"\\boxed\{([A-Za-z])\}", text)
        if m:
            return m.group(1).upper()
        matches = re.findall(r"\b([A-Z])\b", text.upper())
        return matches[-1] if matches else ""

    rewards = []
    for text, gold, options in zip(completions, answers, options_list):
        try:
            # ── MCQ GRADING ──
            if options:
                # Grab the first element of the list (e.g., "C" from ["C"])
                gold_letter = str(gold[0]).strip().upper()
                pred_letter = extract_letter(text)
                
                if pred_letter == gold_letter:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
                    
            # ── FRQ GRADING ──
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5) # Prevent auto_judge from hanging
                try:
                    # Pass the raw text exactly like inference does
                    match = jud.auto_judge(
                        pred=text,
                        gold=gold_list,
                        options=[[]] * len(gold_list),
                    )
                    rewards.append(1.0 if match else -1.0)
                    
                except TimeoutException:
                    rewards.append(-1.0)
                finally:
                    signal.alarm(0)

        # Catch judger crashes just like the inference script does
        except Exception:
            signal.alarm(0)
            rewards.append(-1.0)

    return rewards


def combined_reward(completions, prompts=None, **kwargs) -> List[float]:
    answers      = kwargs.get("answer",  [None] * len(completions))
    options_list = kwargs.get("options", [None] * len(completions))

    fmt  = format_reward_fn(completions, options_list)
    corr = correctness_reward_fn(completions, answers, options_list)

    total = [f + c for f, c in zip(fmt, corr)]

    avg_fmt  = sum(fmt)  / len(fmt)
    avg_corr = sum(corr) / len(corr)
    avg_tot  = sum(total) / len(total)
    print(f"[reward] format={avg_fmt:.3f}  correctness={avg_corr:.3f}  total={avg_tot:.3f}")

    return total


# ─────────────────────────────────────────
# MODAL TRAINING FUNCTION
# ─────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400, # 24-hour timeout to match SFT
    volumes={
        "/checkpoints": CHECKPOINT_VOL,
        "/cache":       CACHE_VOL,
    },
    secrets=[modal.Secret.from_name("hf_token")],
)
def train_grpo(
    train_records: List[Dict[str, Any]],
    hf_repo_id: Optional[str] = None,
    private_hf_repo: bool = False,
    base_model_id: str = BASE_MODEL_ID,
    sft_adapter_id: Optional[str] = None,
    output_dir: str = REMOTE_OUTPUT_DIR,
) -> Dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import GRPOTrainer, GRPOConfig
    import sys, os

    print("in")
    os.chdir("/root")
    sys.path.insert(0, "/root")
    print("in2")

    os.environ["HF_HOME"]                = "/cache/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"]  = "/cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"]     = "/cache/huggingface/transformers"

    print("in3")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    print("base_model_id:     ", base_model_id)
    print("sft_adapter_id:    ", sft_adapter_id)
    print("num_train_records: ", len(train_records))
    print("output_dir:        ", output_dir)
    print("hf_repo_id:        ", hf_repo_id)
    print("cuda_available:    ", torch.cuda.is_available())
    print("cuda device count:   ", torch.cuda.device_count())

    # ── Tokenizer ────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir="/cache/huggingface/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # ── Model (16-bit, no quantization) ──────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir="/cache/huggingface/hub",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False

    # ── Load SFT adapter if provided ─────────────────────────────────────────────
    if sft_adapter_id:
        print(f"Loading SFT adapter from: {sft_adapter_id}")
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

    # ── GRPO Config (vLLM Enabled) ────────────────────────────────────────────────
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
        report_to="none",
        
        # GRPO-specific
        num_generations=NUM_GENERATIONS,
        max_completion_length=GRPO_MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
        
        # vLLM Integration for fast rollouts
        use_vllm=True,
        vllm_max_model_length=MAX_SEQ_LENGTH,
        vllm_gpu_memory_utilization=0.5,
        vllm_mode="colocate",
        
        # Keep memory usage reasonable
        gradient_checkpointing=True,
        optim="adamw_torch", 
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
        processing_class=tokenizer,
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
# LOCAL ENTRYPOINT (Modal CLI Wrapper)
# ─────────────────────────────────────────

@app.local_entrypoint()
def main(
    train_file: str = "rl/grpo_data.jsonl",
    train_file_2: str = None,
    sft_adapter_id: str = "none",
    hf_repo_id: str = "hzia360/qwen3-4b-grpo-lora-hz",
    private_hf_repo: bool = False,
    base_model_id: str = BASE_MODEL_ID,
    output_dir: str = REMOTE_OUTPUT_DIR,
):
    """
    Run locally to dispatch the job to Modal.
    Modal maps these args to CLI flags. 
    Example: modal run grpo.py --train-file data/custom.jsonl --hf-repo-id my/repo
    """
    train_records = load_grpo_records(train_file)

    if train_file_2:
        extra = load_grpo_records(train_file_2)
        train_records = train_records + extra
        print(f"Combined dataset size: {len(train_records)}")

    sft_adapter = None if sft_adapter_id.lower() == "none" else sft_adapter_id

    print(f"Spawning detached Modal task for {len(train_records)} questions...")
    
    # .spawn() ensures this runs in the background and returns terminal control immediately
    train_grpo.spawn(
        train_records=train_records,
        hf_repo_id=hf_repo_id,
        private_hf_repo=private_hf_repo,
        base_model_id=base_model_id,
        sft_adapter_id=sft_adapter,
        output_dir=output_dir,
    )