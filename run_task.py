import argparse
import json
import os
import re
import sys
import wandb
import pandas as pd
from pathlib import Path
from typing import Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from judger import Judger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=bool, default=False, help="Whether this is evaluation (public) or inference (private) run")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit the number of questions to process")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the JSONL results")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    args = parser.parse_args()

    evaluation = args.eval
    # ── Configuration ─────────────────────────────────────────────────────────────
    MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
    GPU_ID      = "0"
    DATA_PATH   = "data/public.jsonl" if evaluation else "data/private.jsonl"
    OUTPUT_PATH = args.output_path
    MAX_TOKENS  = 4096

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


    wandb.init(
        entity="dame-dolla",
        project="cse151b",
        group="exp-02-prompts",
        job_type="evaluate" if evaluation else "inference",
        name="eval-02prompt-200q",
        tags=["prompts", "public-data"],
        config={
            "model_id": MODEL_ID,
            "max_tokens": MAX_TOKENS,
            "dataset": DATA_PATH,
            "temperature": args.temperature,
            "top_p": args.top_p
        }
    )

    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")


    SYSTEM_PROMPT_MATH = (
        "You are an expert mathematician.\n"
        "Use concise symbolic reasoning inside <think> tags. "
        "Focus only on essential equations, derivations, and computations. "
        "Avoid conversational text, repetition, and unnecessary verification.\n"
        "Output the final answer inside a single \\boxed{}.\n"
        "CRITICAL FORMATTING RULES:\n"
        "1. Do not include units or labels inside \\boxed{}.\n"
        "2. If the problem has multiple sub-answers, separate answers with commas inside one box, e.g., \\boxed{3, 7}.\n"
        "3. Preserve required parentheses, brackets, and interval notation, e.g., \\boxed{(2, -2)}.\n"
        "4. Prefer exact symbolic forms. If you are to use decimals, you MUST provide the answer to at least 6 decimal place.\n"
        "5. Do not output anything after the final boxed answer."
    )

    SYSTEM_PROMPT_MCQ = (
        "You are an expert mathematician.\n"
        "Solve the problem using concise internal reasoning inside <think> tags. "
        "Keep reasoning minimal, precise, and computation-focused.\n"
        "Then output ONLY the final multiple-choice answer as a single uppercase letter inside \\boxed{}.\n"
        "CRITICAL FORMATTING RULES:\n"
        "- Output exactly one boxed uppercase letter.\n"
        "- Example: \\boxed{C}\n"
        "- Do not output the answer text or numeric value.\n"
        "- Do not include punctuation inside the box.\n"
        "- Do not output anything after the boxed answer."
    )

    def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for a question."""
        if options:
            labels    = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_MATH, question
    

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        trust_remote_code=True,
        max_num_seqs=16,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("Model loaded.")

    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Using only the first {args.num_samples} questions for evaluation.")

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    responses = [out.outputs[0].text.strip() for out in outputs]


    def extract_letter(text: str) -> str:
        m = re.search(r"\\boxed\{([A-Za-z])\}", text)
        if m:
            return m.group(1).upper()
        matches = re.findall(r"\b([A-Z])\b", text.upper())
        return matches[-1] if matches else ""

    def score_mcq(response: str, gold_letter: str) -> bool:
        return extract_letter(response) == gold_letter.strip().upper()

    sys.path.insert(0, ".")
    judger = Judger(strict_extract=False) 

    results = []

    if evaluation:
        prediction_table = wandb.Table(columns=["ID", "Type", "Ground Truth", "Model Response", "Correct"])
        for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
            is_mcq = bool(item.get("options"))
            gold   = item["answer"]

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
                except Exception as e:
                    print(f"JUDGER CRASHED ON: {item.get("id")} | Error: {e}")
                    correct = False

            results.append({
                "id":       item.get("id"),
                "is_mcq":   is_mcq,
                "gold":     gold,
                "response": response,
                "correct":  correct,
            })
            q_type = "MCQ" if is_mcq else "Free-form"
            prediction_table.add_data(item.get("id"), q_type, str(gold), response, correct)
        
        print(f"Evaluation complete. {len(results)} results.")

        mcq_res  = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]

        def acc(subset):
            return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

        overall_accuracy = acc(results)
        mcq_accuracy = acc(mcq_res)
        free_accuracy = acc(free_res)

        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({mcq_accuracy:.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({free_accuracy:.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({overall_accuracy:.2f}%)")
        print("=" * 50)

        wandb.log({
            "eval/overall_accuracy": overall_accuracy,
            "eval/mcq_accuracy": mcq_accuracy,
            "eval/free_form_accuracy": free_accuracy,
            "predictions": prediction_table
        })
    else:
        prediction_table = wandb.Table(columns=["ID", "Type", "Model Response"])
        for item, response in tqdm(zip(data, responses), total=len(data), desc="Recording"):
            is_mcq = bool(item.get("options"))
            results.append({
                "id":       item.get("id"),
                "is_mcq":   is_mcq,
                "response": response,
            })
            q_type = "MCQ" if is_mcq else "Free-form"
            prediction_table.add_data(item.get("id"), q_type, response)
        print(f"Inference complete. {len(results)} results.")

        wandb.log({
            "predictions": prediction_table
        })

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            if evaluation:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                        "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {out_path}")

    csv_path = None
    if not evaluation:
        csv_path = out_path.with_suffix('.csv')
        df = pd.DataFrame(results)[["id", "response"]]
        df.to_csv(csv_path, index=False)
        print(f"Generated submission CSV: {csv_path}")

    artifact_name = f"results-{wandb.run.id}"
    artifact_type = "evaluation_results" if evaluation else "submission"
    
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(str(out_path))
    if csv_path:
        artifact.add_file(str(csv_path))

    wandb.log_artifact(artifact)
    print("Files successfully uploaded to Weights & Biases Artifacts.")
    
    wandb.finish()

if __name__ == "__main__":
    main()