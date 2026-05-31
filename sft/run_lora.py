import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507" 
DATASET_FILE = "/workspace/data/sft_data1.jsonl"
OUTPUT_DIR = "/workspace/results/lora_train/math1"

MAX_SEQ_LENGTH = 12288
LORA_RANK = 32 
LEARNING_RATE = 2e-5
EPOCHS = 3

print(f"Loading Base Model: {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = torch.bfloat16,
    load_in_4bit = False,
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
    random_state = 3407,
    trust_remote_code = True
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
        num_train_epochs = EPOCHS,
        learning_rate = LEARNING_RATE,
        fp16 = False,
        bf16 = True,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
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