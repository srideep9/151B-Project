import json
import re
import sys
from tqdm import tqdm

sys.path.insert(0, ".")
from judger import Judger

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()

def main():
    INPUT_FILE = "distillation/fixed_formatting.jsonl"
    PUBLIC_FILE = "data/public.jsonl"
    OUTPUT_TRAIN_FILE = "distillation/training_ready.jsonl"
    
    with open(PUBLIC_FILE, 'r') as qf:
        questions = {obj["id"]: obj for obj in map(json.loads, qf)}

    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    judger = Judger(strict_extract=False)
    results = []
    training_data = []

    for item in tqdm(data, desc="Scoring & Filtering"):
        is_mcq = item.get("is_mcq", False)
        gold = item.get("gold")
        response = item.get("response", "")
        q_id = item.get("id")

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append({
            "id": q_id,
            "is_mcq": is_mcq,
            "correct": correct
        })

        if correct:
            orig_q = questions[q_id]
            training_data.append({
                "id": q_id,
                "is_mcq": is_mcq,
                "question": orig_q.get("question"),
                "options": orig_q.get("options"), 
                "response": response
            })

    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("\n" + "=" * 50)
    print("UPDATED EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)

    with open(OUTPUT_TRAIN_FILE, 'w') as f:
        for row in training_data:
            f.write(json.dumps(row) + "\n")
            
    print(f"Saved {len(training_data)} correct responses to {OUTPUT_TRAIN_FILE} for SFT.")

if __name__ == "__main__":
    main()