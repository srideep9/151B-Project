import json
import re
import os
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT_OPENTHOUGHTS = (
    "You are an MIT mathematician.\n"
    "Solve the problem by thinking step-by-step inside <think> tags. "
    "Explore multiple paths if necessary to guarantee the correct mathematical result.\n"
    "Then output the final answer inside a single \\boxed{}."
)

SYSTEM_PROMPT_MATH = (
    "You are an MIT mathematician.\n"
    "Solve the problem using extremely concise internal reasoning inside <think> tags. "
    "Keep reasoning precise and computation-focused. "
    "Avoid any conversational text, repetition, and unnecessary verification. No explanations or narrative sentences.\n"
    "Then output the final answer(s) inside a single \\boxed{}.\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Do not include units or labels inside \\boxed{}.\n"
    "- If the problem has multiple [ANS] blanks, output the answers in the exact order they are requested, separated by commas, inside one box. The number of comma-separated items inside your \\boxed{} MUST exactly match the number of [ANS] placeholders in the question.\n"
    "- If there is only ONE [ANS] placeholder, but the solution has multiple values, you MUST group them inside parentheses. Correct: \\boxed{(7, -7)}.\n"
    "- Always prefer exact symbolic forms for answers. Do not convert fractions to decimals. If a decimal is required, you MUST provide the answer to at least 6 decimal places. Never round or truncate intermediate values. Carry full precision through every step.\n"
    "- NEVER debate or second-guess formatting expectations inside the <think> tags. Once derived, immediately output the \\boxed{} and stop.\n"
    "- Do not output anything after the boxed answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an MIT mathematician.\n"
    "Solve the problem using extremely concise internal reasoning inside <think> tags. "
    "Keep reasoning precise and computation-focused. "
    "Avoid any conversational text, repetition, and unnecessary verification. No explanations or narrative sentences.\n"
    "Then output ONLY the final multiple-choice answer as a single uppercase letter inside a \\boxed{}.\n"
    "CRITICAL FORMATTING RULES:\n"
    "- Output exactly one boxed uppercase letter as your answer, e.g., \\boxed{C}\n"
    "- Do not output the answer text or numeric value.\n"
    "- Do not include punctuation inside the box.\n"
    "- NEVER debate or second-guess formatting expectations inside the <think> tags.\n"
    "- Do not output anything after the boxed answer."
)

def create_message_format(system_prompt, user_question, assistant_response):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": assistant_response}
        ]
    }

def build_dataset():
    final_dataset = []
    
    # ==========================================
    # STEP 1: Process the Verified Math Dataset
    # ==========================================
    print("Downloading and cleaning Verified OpenThoughts Math...")
    ot_dataset = load_dataset("open-r1/OpenThoughts-114k-math", split="train") 
    ot_count = 0
    
    for row in tqdm(ot_dataset, desc="Processing Verified OpenThoughts"):
        if not row.get("correct", False):
            continue
            
        token_count = row.get("generated_token_count", float('inf'))
        if token_count >= 8000:
            continue
            
        question = row.get("problem", "")
        conversations = row.get("conversations", [])
        if not conversations:
            continue
        raw_response = conversations[-1].get("value", "")
        
        think_match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', raw_response, flags=re.DOTALL)
        if not think_match:
            continue
        thought_process = think_match.group(1).strip()
        
        solution_block = raw_response.split("<|end_of_thought|>")[-1]
        box_contents = re.findall(r'\\boxed\{([^}]*)\}', solution_block)
        
        if not box_contents:
            continue
            
        merged_box = f"\\boxed{{{', '.join(item.strip() for item in box_contents)}}}"
        clean_response = f"<think>\n{thought_process}\n</think>\n{merged_box}"
        
        # USE THE RELAXED PROMPT FOR OPEN-SOURCE DATA
        final_dataset.append(create_message_format(SYSTEM_PROMPT_OPENTHOUGHTS, question, clean_response))
        ot_count += 1
        
        if ot_count >= 10000:
            break

    # ==========================================
    # STEP 2: Process Your Golden Rows (Dynamic)
    # ==========================================
    print("Loading and oversampling Kaggle Golden Rows...")
    golden_rows = []
    with open("distillation/training_ready.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            question = data.get("question", "")
            response = data.get("response", "")
            is_mcq = data.get("is_mcq", False)
            
            # USE THE STRICT KAGGLE PROMPTS FOR YOUR DATA
            selected_prompt = SYSTEM_PROMPT_MCQ if is_mcq else SYSTEM_PROMPT_MATH
            
            if is_mcq:
                options = data.get("options")
                if options:
                    question += "\nOptions:\n" + "\n".join(options)
                
            golden_rows.append(create_message_format(selected_prompt, question, response))
    
    # OVERSAMPLING: Multiply by 7
    oversample_multiplier = 7
    final_dataset.extend(golden_rows * oversample_multiplier)
    
    # ==========================================
    # STEP 3: Save Final SFT File
    # ==========================================
    import random
    random.shuffle(final_dataset)
    
    with open("sft/final_sft_training_data.jsonl", "w") as f:
        for row in final_dataset:
            f.write(json.dumps(row) + "\n")
            
    print(f"\nSUCCESS! Generated final_sft_training_data.jsonl")
    print(f"Total Rows: {len(final_dataset)} ({ot_count} OpenThoughts + {len(golden_rows)*oversample_multiplier} Kaggle)")

if __name__ == "__main__":
    os.environ["HF_TOKEN"] = "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"
    build_dataset()