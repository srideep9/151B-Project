import sys
import os

import modal

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507" 
DATASET_FILE = "final_sft_training_data.jsonl"
OUTPUT_DIR = "/results/math2"

MAX_SEQ_LENGTH = 12288
LORA_RANK = 32 
LEARNING_RATE = 2e-5
EPOCHS = 3

app = modal.App("unsloth-sft-h100")
vol = modal.Volume.from_name("sft-results", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel")
    .env({
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })
    .pip_install(
        "unsloth",
        "trl",
        "bitsandbytes",
        "pandas", 
        "tqdm",
        "numpy",
        "antlr4-python3-runtime==4.11.1",
        "sympy", # Often required by math judgers
        "huggingface_hub",
        "hf_transfer"
    )
    .run_commands("echo 'Busting cache to fix vllm lora models error'")
    .add_local_file("sft/final_sft_training_data.jsonl", remote_path="/root/final_sft_training_data.jsonl")
)

@app.function(
    image=image,
    gpu="H100:1", # Requesting exactly 2 H200s
    timeout=86400,             # 24-hour timeout to prevent premature kill
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def sft():
    import torch
    import unsloth
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    os.chdir("/root")
    sys.path.insert(0, "/root")
    
    print(f"Loading Base Model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = torch.bfloat16,
        load_in_4bit = False,
        trust_remote_code = True
    )

    print("Injecting LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = LORA_RANK * 2,
        lora_dropout = 0,
        bias = "none",    
        use_gradient_checkpointing = "unsloth",
        random_state = 3407
    )

    print("Loading and formatting dataset...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    if tokenizer.chat_template is not None:
        print("Native chat template detected. Using model's exact pre-training format.")
        tokenizer = get_chat_template(
            tokenizer,
            mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}
        )
    else:
        print("Warning: No native template found. Falling back to generic ChatML.")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "chatml",
            mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}
        )

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 8,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_ratio = 0.05,
            num_train_epochs = 2,
            learning_rate = LEARNING_RATE,
            fp16 = False,
            bf16 = True,
            logging_steps = 10,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            save_strategy = "epoch",
        ),
    )

    print("Starting the A100 Engines...")
    trainer_stats = trainer.train()

    print("Training Complete. Saving Final Adapter...")
    model.save_pretrained(f"{OUTPUT_DIR}_final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}_final")

    print("Run successful. Adapter weights are ready for the Kaggle vLLM pipeline.")

@app.local_entrypoint()
def main():
    sft.spawn()