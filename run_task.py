import json
import os
import re
import sys
import signal
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import Optional

import modal

# ── Modal Configuration ───────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
app = modal.App("vllm-inference-h200")

vol = modal.Volume.from_name("inference-results", create_if_missing=True)

def download_model():
    """Bakes the model weights into the Modal image during deployment."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID)

# Define the environment, install dependencies, and cache the model
image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel")
    .env({
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })
    .pip_install(
        "vllm", 
        "transformers", 
        "pandas", 
        "tqdm",
        "numpy",
        "antlr4-python3-runtime==4.11.1",
        "sympy", # Often required by math judgers
        "huggingface_hub",
        "hf_transfer"
    )
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")]
        )
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("judger.py", remote_path="/root/judger.py")
    .add_local_file("prompt_strategy.py", remote_path="/root/prompt_strategy.py")
    .add_local_dir("data", remote_path="/root/data")
)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Sympy evaluation took too long.")

def evaluate_single_item(item, response):
    """Worker function to evaluate a single question on a separate CPU core."""
    import re
    from judger import Judger
    
    # Initialize a fresh judger for this specific CPU process
    judger = Judger(strict_extract=False)
    
    def extract_letter(text: str) -> str:
        m = re.search(r"\\boxed\{([A-Za-z])\}", text)
        if m: return m.group(1).upper()
        matches = re.findall(r"\b([A-Z])\b", text.upper())
        return matches[-1] if matches else ""

    is_mcq = bool(item.get("options"))
    gold = item["answer"]
    correct = False

    if is_mcq:
        correct = extract_letter(response) == str(gold).strip().upper()
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

    return {
        "id": item.get("id"),
        "is_mcq": is_mcq,
        "gold": gold,
        "response": response,
        "correct": correct,
    }

def process_single_majority_vote(generated_texts: list[str]) -> str:
    """Worker function to process majority voting for a single question on a separate CPU core."""
    import signal
    from judger import Judger

    class TimeoutException(Exception): pass
    def timeout_handler(signum, frame): raise TimeoutException()

    judger = Judger(strict_extract=False)

    def compare_predictions(pred1: str, pred2: str, timeout_sec=4) -> bool:
        # SUPER SPEED HACK: If the strings match exactly (ignoring spaces), skip Sympy!
        if pred1.replace(" ", "") == pred2.replace(" ", ""):
            return True

        list1 = judger.split_by_comma(pred1)
        list2 = judger.split_by_comma(pred2)
        
        if len(list1) != len(list2): return False
            
        for item1, item2 in zip(list1, list2):
            norm1 = judger.norm_ans_str(item1)
            norm2 = judger.norm_ans_str(item2)

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)

            try:
                is_match = judger.is_equal(norm1, norm2)
            except TimeoutException:
                is_match = False
            except Exception:
                is_match = False
            finally:
                signal.alarm(0)

            if not is_match: return False
                
        return True

    extracted_data = []
    for resp in generated_texts:
        ans = judger.extract_ans(resp)
        if ans:
            extracted_data.append({"raw_extracted": ans, "raw_response": resp})
                
    if not extracted_data:
        return generated_texts[0]
        
    equivalence_groups = []
    for item in extracted_data:
        ans = item["raw_extracted"]
        raw_text = item["raw_response"]
        matched = False
        
        for group in equivalence_groups:
            if compare_predictions(ans, group["representative"]):
                group["count"] += 1
                group["raw_responses"].append(raw_text)
                matched = True
                break
                
        if not matched:
            equivalence_groups.append({
                "representative": ans, "count": 1, "raw_responses": [raw_text]
            })
            
    best_group = max(equivalence_groups, key=lambda x: (x["count"], -len(min(x["raw_responses"], key=len))))
    return min(best_group["raw_responses"], key=len)

# ── Remote GPU Execution ──────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="H200:2",
    timeout=86400,           
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_and_evaluate(args: dict):
    os.chdir("/root")
    sys.path.insert(0, "/root")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from judger import Judger

    from prompt_strategy import (
        build_prompt,
        classify_math_topic,
        select_curated_examples,
        select_few_shot_examples,
    )

    evaluation = args["eval"]
    DATA_PATH = "data/public.jsonl" if evaluation else "data/private.jsonl"
    MAX_TOKENS = args["max_tokens"]

    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # vLLM will automatically partition across the 2 allocated H200s
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=2,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_model_len=MAX_TOKENS + 4096,
        trust_remote_code=True,
        max_num_seqs=args["max_num_seqs"],
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=args["temperature"],
        top_p=args["top_p"],
        top_k=20,
        min_p=0.0,
        n=args["num_outputs"],
        presence_penalty=0.0,
        repetition_penalty=1.05,
    )

    print("Model loaded.")



    if args["num_samples"] is not None:
        data = data[:args["num_samples"]]
        print(f"Using only the first {args['num_samples']} questions.")

    messages_list = []
    prompt_records = []
    prompt_mode = args["prompt_mode"]
    example_source = args["example_source"]
    shots = int(args["shots"])

    for item in data:
        topic = classify_math_topic(item["question"])
        examples = []
        if prompt_mode == "routed" and example_source != "none":
            is_mcq = bool(item.get("options"))
            if example_source in ("curated", "mixed"):
                examples.extend(select_curated_examples(topic.name, shots, is_mcq))
            if example_source == "dataset" or (
                example_source == "mixed" and len(examples) < shots
            ):
                examples.extend(select_few_shot_examples(data, item, shots - len(examples)))

        system_prompt, user_prompt = build_prompt(
            item["question"],
            item.get("options"),
            topic=topic,
            few_shot_examples=examples,
            include_topic_hint=prompt_mode == "routed",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages_list.append(messages)
        prompt_records.append(
            {
                "id": item.get("id"),
                "topic": topic.name,
                "is_mcq": bool(item.get("options")),
                "few_shot_ids": [example.get("id") for example in examples],
            }
        )

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    print("Generation complete.")

    print("Running Parallel Mathematical Majority Voting...")
    responses = []
    
    # 1. Prepare the data (extract the raw text lists from vLLM outputs)
    all_generated_texts = [
        [comp.text.strip() for comp in out.outputs] 
        for out in outputs
    ]

    # 2. Spin up the CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the worker function to our list of 31-generation text blocks
        results_iterator = executor.map(process_single_majority_vote, all_generated_texts)
        
        # Wrap in tqdm so we can watch it fly
        for winning_trace in tqdm(results_iterator, total=len(all_generated_texts), desc="Voting"):
            responses.append(winning_trace)
    print("Voting complete!")

    results = []
    if evaluation:
        print("Starting parallel scoring across all CPU cores...")
        
        # Spin up a pool of workers using all available CPU cores on the machine
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Send all the items and responses to the workers
            results_iterator = executor.map(
                evaluate_single_item, 
                data, 
                responses
            )
            
            # Wrap in tqdm to watch the progress bar fly
            for result in tqdm(results_iterator, total=len(data), desc="Scoring"):
                results.append(result)
    else:
        for item, response in tqdm(zip(data, responses), total=len(data), desc="Recording"):
            is_mcq = bool(item.get("options"))
            results.append({
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "response": response,
            })

    if evaluation:
        mcq_res  = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]

        def acc(subset):
            return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
        print("=" * 50)
    
    out_path = Path(args["output_path"])
    with open(out_path, "w") as f:
        for r in results:
            if evaluation:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                          "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")
    vol.commit()

    csv_path = None
    if not evaluation:
        csv_path = out_path.with_suffix('.csv')
        df = pd.DataFrame(results)[["id", "response"]]
        df.to_csv(csv_path, index=False)
        print(f"Generated submission CSV: {csv_path}")

    print(f"Saved {len(results)} records to {out_path}")

    print("Teacher generation complete!")

# ── Local Entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    eval: bool = True,
    num_samples: int = None,
    num_outputs: int = 31,
    output_path: str = "/results/inference.jsonl",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    max_num_seqs: int = 128,
    prompt_mode: str = "routed",
    example_source: str = "curated",
    shots: int = 1
):
    """
    Modal natively handles the CLI argument parsing for these parameters.
    Example: modal run modal_distillation.py --eval --output-path results.jsonl
    """
    args_dict = {
        "eval": eval,
        "num_samples": num_samples,
        "num_outputs": num_outputs,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_num_seqs": max_num_seqs,
        "output_path": output_path,
        "prompt_mode": prompt_mode,
        "example_source": example_source,
        "shots": shots,
    }

    print(f"Triggering Remote H200 Generation. Results will be saved on volume to: {output_path}")
    
    # Trigger the remote function
    generate_and_evaluate.spawn(args_dict)