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
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
app = modal.App("vllm-distillation-h200")

vol = modal.Volume.from_name("distillation-results", create_if_missing=True)

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
    gpu="H200:2", # Requesting exactly 2 H200s
    timeout=86400,             # 24-hour timeout to prevent premature kill
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

    evaluation = args["eval"]
    DATA_PATH = "data/public.jsonl" if evaluation else "data/private.jsonl"
    MAX_TOKENS = args["max_tokens"]

    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

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

    few_shot_history_math = [
        {"role": "system", "content": SYSTEM_PROMPT_MATH},
        {"role": "user", "content": "Evaluate the definite integral $\\int_0^1 x e^x \\, dx$."},
        {"role": "assistant", "content": r"""<think>
Use integration by parts: \int u \, dv = uv - \int v \, du.
Let u = x, dv = e^x dx.
Then du = dx, v = e^x.
\int x e^x dx = x e^x - \int e^x dx = x e^x - e^x.
Evaluate from 0 to 1:
Upper bound (x=1): 1 \cdot e^1 - e^1 = e - e = 0.
Lower bound (x=0): 0 \cdot e^0 - e^0 = 0 - 1 = -1.
Result: 0 - (-1) = 1.
</think>
\boxed{1}"""},
        {"role": "user", "content": "Given the real part $u(x, y) = e^x \\cos(y)$ of an analytic function $f(z) = u + \\mathrm{i} v$, find $f(z)$ in terms of $z$ if $f(0) = 1$."},
        {"role": "assistant", "content": r"""<think>
Use Milne-Thomson method: f'(z) = u_x(z, 0) - \mathrm{i} u_y(z, 0).
u(x, y) = e^x \cos(y)
u_x(x, y) = e^x \cos(y) \implies u_x(z, 0) = e^z \cos(0) = e^z
u_y(x, y) = -e^x \sin(y) \implies u_y(z, 0) = -e^z \sin(0) = 0
f'(z) = e^z - \mathrm{i}(0) = e^z
Integrate with respect to z: f(z) = e^z + C.
Apply condition: f(0) = 1 \implies e^0 + C = 1 \implies C = 0.
f(z) = e^z.
</think>
\boxed{e^z}"""},
        {"role": "user", "content": "A test has $H_0: \\mu = 50$ vs $H_a: \\mu < 50$. Sample size $n=16$, sample mean $48$, standard deviation $4$. Find: A. Test statistic. B. Rejection region for $\\alpha=0.05$."},
        {"role": "assistant", "content": r"""<think>
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
t = \frac{48 - 50}{4 / \sqrt{16}} = \frac{-2}{1} = -2.
df = 15. Left-tailed test, \alpha = 0.05. Critical t_{0.05, 15} = -1.753.
Rejection region: (-\infty, -1.753).
</think>
\boxed{-2, (-\infty, -1.753)}"""}
    ]

    few_shot_history_mcq = [
        {"role": "system", "content": SYSTEM_PROMPT_MCQ},
        {"role": "user", "content": "Evaluate the contour integral $\\oint_{|z|=2} \\frac{e^z}{z(z-1)^2} \\mathrm{d}z$ where the contour is oriented counterclockwise.\n\nOptions:\nA. $$2\\pi i e$$\nB. $$2\\pi i (e - 1)$$\nC. $$2\\pi i$$\nD. $$0$$\nE. $$-2\\pi i$$\nF. $$4\\pi i e$$\nG. $$\\pi i$$\nH. $$\\pi i e$$\nI. $$2\\pi i (e+1)$$\nJ. $$-2\\pi i e$$"},
        {"role": "assistant", "content": r"""<think>
Identify singularities: simple pole at z = 0, and a pole of order 2 at z = 1.
Both poles lie inside the contour |z| = 2.
Residue at z = 0: \lim_{z \to 0} z \frac{e^z}{z(z-1)^2} = \frac{e^0}{(-1)^2} = 1.
Residue at z = 1: \lim_{z \to 1} \frac{d}{dz} \left[ (z-1)^2 \frac{e^z}{z(z-1)^2} \right] = \lim_{z \to 1} \frac{d}{dz} \left( \frac{e^z}{z} \right).
Derivative using quotient rule: \frac{z e^z - e^z}{z^2}.
Evaluate at z = 1: \frac{1 \cdot e^1 - e^1}{1^2} = 0.
Sum of residues: 1 + 0 = 1.
By Cauchy's Residue Theorem: \oint = 2\pi i \sum \text{Res} = 2\pi i (1) = 2\pi i.
Matches option C.
</think>
\boxed{C}"""},
        {"role": "user", "content": "Find the family of orthogonal trajectories to the family of parabolas $y = c x^2$.\n\nOptions:\nA. $$x^2 - 2y^2 = C$$\nB. $$x^2 + y^2 = C$$\nC. $$x^2 + 2y^2 = C$$\nD. $$2x^2 + y^2 = C$$\nE. $$x^2 + 4y^2 = C$$\nF. $$x^2 - y^2 = C$$\nG. $$2x^2 - y^2 = C$$\nH. $$x^2 + y = C$$\nI. $$x + y^2 = C$$\nJ. $$x^2 y = C$$"},
        {"role": "assistant", "content": r"""<think>
Differentiate y = cx^2 with respect to x: y' = 2cx.
Eliminate the parameter c using the original equation c = y/x^2.
Substitute c into the derivative: y' = 2(y/x^2)x = \frac{2y}{x}.
The differential equation of the original family is y' = \frac{2y}{x}.
For the orthogonal trajectories, replace y' with -1/y'.
\frac{-1}{y'} = \frac{2y}{x} \implies y' = \frac{-x}{2y}.
Separate variables: 2y dy = -x dx.
Integrate both sides: \int 2y dy = \int -x dx \implies y^2 = -\frac{x^2}{2} + K.
Multiply by 2: 2y^2 = -x^2 + 2K \implies x^2 + 2y^2 = C.
Matches option C.
</think>
\boxed{C}"""},
        {"role": "user", "content": "We now define an algorithm: The definition of a(n) is the coordination sequence for a specific lattice. Given the input x_list: [1, 2, 3], determine the corresponding output sequence y_list.\n\nOptions:\nA. [4, 8, 12]\nB. [4, 9, 14]\nC. [6, 12, 18]\nD. [6, 18, 38]\nE. [8, 16, 24]"},
        {"role": "assistant", "content": r"""<think>
The problem asks to match the specific algorithm's output sequence to the given options for x = 1, 2, 3.
By analyzing the sequence generation properties for the requested lattice, the values grow sequentially as 4, 8, 12.
Comparing this derived array [4, 8, 12] to the given options.
Matches option A.
</think>
\boxed{A}"""}
    ]

    def build_messages(question: str, options: Optional[list]) -> list:
        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            user_text = f"{question}\n\nOptions:\n{opts_text}"
            messages = list(few_shot_history_mcq)
            messages.append({"role": "user", "content": user_text})
            return messages
        else:
            messages = list(few_shot_history_math)
            messages.append({"role": "user", "content": question})
            return messages

    prompts = []
    for item in data:
        messages = build_messages(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

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

    print(f"Saved {len(results)} records to {out_path}")

    print("Teacher generation complete!")

# ── Local Entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    eval: bool = True,
    num_samples: int = None,
    num_outputs: int = 31,
    output_path: str = "/results/deepseek_r1_qwen32b_public.jsonl",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    max_num_seqs: int = 128
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
        "output_path": output_path
    }

    print(f"Triggering Remote H200 Generation. Results will be saved on volume to: {output_path}")
    
    # Trigger the remote function
    generate_and_evaluate.spawn(args_dict)