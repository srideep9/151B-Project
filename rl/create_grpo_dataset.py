import json
import os
import re
import random
from datasets import load_dataset
from tqdm import tqdm

def split_safe_comma(text):
    """Splits a string by commas, ignoring commas inside (), [], or {}."""
    parts = []
    current = []
    depth = 0
    
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            
        if char == ',' and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
            
    parts.append("".join(current).strip())
    return [p for p in parts if p]

def extract_boxed_content(text):
    """
    Isolates the solution block and extracts the final boxed answers, 
    safely splitting comma-separated lists into JSON arrays.
    """
    if "<|end_of_thought|>" in text:
        solution_block = text.split("<|end_of_thought|>")[-1]
    elif "</think>" in text:
        solution_block = text.split("</think>")[-1]
    else:
        solution_block = text
        
    raw_answers = []
    start = 0
    
    while True:
        idx = solution_block.find("\\boxed{", start)
        if idx == -1:
            break
            
        content_start = idx + 7 
        depth = 1
        i = content_start
        
        while i < len(solution_block) and depth > 0:
            if solution_block[i] == '{':
                depth += 1
            elif solution_block[i] == '}':
                depth -= 1
            i += 1
            
        if depth == 0:
            raw_answers.append(solution_block[content_start:i-1].strip())
            
        start = i 
        
    if not raw_answers:
        return None
        
    # CRITICAL FIX: Flatten and safely split all extracted answers
    final_answers = []
    for ans in raw_answers:
        final_answers.extend(split_safe_comma(ans))
        
    return final_answers

def build_grpo_dataset():
    grpo_dataset = []
    
    # ==========================================
    # STEP 1: Process UNSEEN OpenThoughts Data
    # ==========================================
    print("Downloading OpenThoughts Math...")
    ot_dataset = load_dataset("open-r1/OpenThoughts-114k-math", split="train") 
    
    ot_count = 0
    start_index = 30000 
    target_count = 8000 
    
    for i in tqdm(range(start_index, len(ot_dataset)), desc="Processing Unseen OpenThoughts"):
        row = ot_dataset[i]
        
        if not row.get("correct", False):
            continue
            
        question = row.get("problem", "")
        conversations = row.get("conversations", [])
        
        if not question or not conversations:
            continue
            
        raw_response = conversations[-1].get("value", "")
        answer = extract_boxed_content(raw_response)
        
        if not answer:
            continue
            
        grpo_dataset.append({
            "question": question,
            "answer": answer,
            "options": None 
        })
        
        ot_count += 1
        if ot_count >= target_count:
            break

    # ==========================================
    # STEP 2: The Golden Holdout (Local Validation)
    # ==========================================
    # Slice off 500 rows before we mix in the Kaggle data
    train_set = grpo_dataset

    # ==========================================
    # STEP 3: Process the FULL Kaggle Public Set
    # ==========================================
    print("Loading and oversampling the full Kaggle Public Set...")
    kaggle_rows = []
    
    # UPDATE THIS PATH: Point this to your raw Kaggle public dataset file
    # Ensure this file contains ALL questions, even the ones the teacher model failed on.
    with open("data/public.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            kaggle_rows.append({
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "options": data.get("options", None)
            })
            
    # OVERSAMPLING: Multiply the Kaggle rows by 5 to anchor the RL algorithm
    oversample_multiplier = 3
    train_set.extend(kaggle_rows * oversample_multiplier)
    
    # ==========================================
    # STEP 4: Save Final GRPO File
    # ==========================================
    # Shuffle one last time so the Kaggle rows are evenly distributed
    random.shuffle(train_set)
    
    with open("rl/grpo_data.jsonl", "w") as f:
        for row in train_set:
            f.write(json.dumps(row) + "\n")
            
    print(f"\nSUCCESS! Generated open_thoughts_grpo.jsonl")
    print(f"Total GRPO Training Rows: {len(train_set)} ({len(train_set) - (len(kaggle_rows)*oversample_multiplier)} OpenThoughts + {len(kaggle_rows)*oversample_multiplier} Kaggle)")

if __name__ == "__main__":
    os.environ["HF_TOKEN"] = "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"
    build_grpo_dataset()