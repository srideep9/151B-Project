#!/usr/bin/env python3
"""
Single-entry inference script for the CSE 151B math competition.

run_inference() loads Qwen/Qwen3-4B-Thinking-2507, runs inference on the private
set by default, applies mathematical majority voting, and writes the final
submission CSV. This is the Modal/W&B-free version of the submitted pipeline.
"""

import argparse
import json
import os
import re
import signal
from pathlib import Path
from typing import Optional

import pandas as pd


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Sympy evaluation took too long.")


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


FEW_SHOT_HISTORY_MATH = [
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
\boxed{-2, (-\infty, -1.753)}"""},
]

FEW_SHOT_HISTORY_MCQ = [
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
\boxed{A}"""},
]


def build_messages(question: str, options: Optional[list]) -> list[dict]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options)
        )
        messages = list(FEW_SHOT_HISTORY_MCQ)
        messages.append({"role": "user", "content": f"{question}\n\nOptions:\n{opts_text}"})
        return messages

    messages = list(FEW_SHOT_HISTORY_MATH)
    messages.append({"role": "user", "content": question})
    return messages


def get_mathematical_majority_vote(generated_texts: list[str]) -> str:
    from judger import Judger

    judger = Judger(strict_extract=False)

    def compare_predictions(pred1: str, pred2: str, timeout_sec=4) -> bool:
        list1 = judger.split_by_comma(pred1)
        list2 = judger.split_by_comma(pred2)
        if len(list1) != len(list2):
            return False

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

            if not is_match:
                return False

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
            equivalence_groups.append(
                {"representative": ans, "count": 1, "raw_responses": [raw_text]}
            )

    best_group = max(
        equivalence_groups,
        key=lambda x: (x["count"], -len(min(x["raw_responses"], key=len))),
    )
    return min(best_group["raw_responses"], key=len)


def evaluate_single_item(item: dict, response: str) -> dict:
    from judger import Judger

    judger = Judger(strict_extract=False)

    def extract_letter(text: str) -> str:
        m = re.search(r"\\boxed\{([A-Za-z])\}", text)
        if m:
            return m.group(1).upper()
        matches = re.findall(r"\b([A-Z])\b", text.upper())
        return matches[-1] if matches else ""

    is_mcq = bool(item.get("options"))
    gold = item["answer"]
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


def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "results/inference.jsonl",
    csv_path: Optional[str] = None,
    model_id: str = MODEL_ID,
    eval: bool = False,
    num_samples: Optional[int] = None,
    num_outputs: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.9,
    cuda_visible_devices: Optional[str] = "0",
) -> str:
    """Run the full inference pipeline and return the submission CSV path."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    data = [json.loads(line) for line in data_file.open()]
    if num_samples is not None:
        data = data[:num_samples]
        print(f"Using only the first {num_samples} questions.")

    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_tokens + 4096,
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        min_p=0.0,
        n=num_outputs,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("Model loaded.")

    prompts = []
    for item in tqdm(data, desc="Rendering prompts"):
        prompts.append(
            tokenizer.apply_chat_template(
                build_messages(item["question"], item.get("options")),
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    print("Generation complete.")

    print("Running Mathematical Majority Voting...")
    responses = []
    for out in tqdm(outputs, desc="Voting"):
        generated_texts = [comp.text.strip() for comp in out.outputs]
        responses.append(get_mathematical_majority_vote(generated_texts))
    print("Voting complete!")

    if eval:
        results = [
            evaluate_single_item(item, response)
            for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring")
        ]
        mcq_res = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]

        def acc(subset: list[dict]) -> float:
            return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
        print("=" * 50)
    else:
        results = [
            {"id": item.get("id"), "is_mcq": bool(item.get("options")), "response": response}
            for item, response in tqdm(zip(data, responses), total=len(data), desc="Recording")
        ]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            if eval:
                record = {
                    "id": r["id"],
                    "is_mcq": r["is_mcq"],
                    "gold": r["gold"],
                    "response": r["response"],
                    "correct": r["correct"],
                }
            else:
                record = {
                    "id": r["id"],
                    "is_mcq": r["is_mcq"],
                    "response": r["response"],
                }
            f.write(json.dumps(record) + "\n")

    final_csv_path = Path(csv_path) if csv_path else out_path.with_suffix(".csv")
    final_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results)[["id", "response"]].to_csv(final_csv_path, index=False)

    print(f"Generated submission CSV: {final_csv_path}")
    print(f"Saved {len(results)} records to {out_path}")
    return str(final_csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run local vLLM inference.")
    parser.add_argument("--data-path", default="data/private.jsonl")
    parser.add_argument("--output-path", default="results/inference.jsonl")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-outputs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        data_path=args.data_path,
        output_path=args.output_path,
        csv_path=args.csv_path,
        model_id=args.model_id,
        eval=args.eval,
        num_samples=args.num_samples,
        num_outputs=args.num_outputs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cuda_visible_devices=args.cuda_visible_devices,
    )


if __name__ == "__main__":
    main()
