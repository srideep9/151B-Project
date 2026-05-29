"""
Generate teacher reasoning traces on Modal (GPU).

Usage:
  - Edit constants at top as needed (MODEL_ID, OUTPUT_FILE, sampling params).
  - Ensure you have configured Modal locally and added any secrets (HF_TOKEN) to your Modal profile.
  - Run this script locally to launch the Modal job (it will NOT auto-run on import).

This script creates a Modal function that runs `vllm` on a GPU-enabled container to
produce N traces per example, returns them to the local runner which performs
majority voting, checkpointing, and incremental saving to JSONL.

Notes / TODOs for runtime:
  - The Docker image used here is pulled from Docker Hub and then `pip install`
    is used to install Python deps. Building that image may take some time.
  - You must add your Hugging Face token to Modal as a secret named `HF_TOKEN`.
  - If you prefer a prebuilt image with vLLM + CUDA binary wheels, replace
    `BASE_IMAGE` below with the working image name.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Sympy evaluation took too long.")


def disable_optional_vllm_kernels() -> None:
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_DISABLED_KERNEL_WARMUP"] = "1"
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"


disable_optional_vllm_kernels()

import modal

# -----------------------------
# USER-CONFIGURABLE CONSTANTS
# -----------------------------
MODEL_ID = "Qwen/QwQ-32B"  # Teacher
DATA_PATH = "data/private.jsonl"
OUTPUT_FILE = "data/private_teacher_traces_qwq32b_modal.jsonl"

# Sampling / generation params (easily editable)
N_GEN = 3  # generations per question
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 2048
BATCH_SIZE = 1  # number of questions to send per Modal call
MAX_NUM_SEQS = 3
MAX_MODEL_LEN = 8192

# Modal settings
MODAL_STUB_NAME = "qwq32b-traces"
# Qwen/QwQ-32B requires a large GPU. Use H200 for Modal when available.
GPU_TYPE = "H200"
BASE_IMAGE = "nvidia/cuda:12.8.2-runtime-ubuntu22.04"  # valid CUDA 12.8 tag with runtime support and Python install via Modal
VLLM_IMAGE_PACKAGES = [
    "vllm==0.11.0",
    "transformers==4.56.2",
    "tokenizers==0.22.1",
    "sentencepiece",
    "numpy",
    "sympy",
    "tqdm",
]

# System prompts and few-shot histories from run_task.py.
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

few_shot_history_math = [
    {"role": "system", "content": SYSTEM_PROMPT_MATH},
    {"role": "user", "content": "Evaluate the definite integral $\\int_0^1 x e^x \\,, dx$."},
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
\\boxed{1}"""},
    {"role": "user", "content": "Given the real part $u(x, y) = e^x \\cos(y)$ of an analytic function $f(z) = u + \\mathrm{i} v$, find $f(z)$ in terms of $z$ if $f(0) = 1$."},
    {"role": "assistant", "content": r"""<think>
Use Milne-Thomson method: f'(z) = u_x(z, 0) - \\mathrm{i} u_y(z, 0).
u(x, y) = e^x \\cos(y)
u_x(x, y) = e^x \\cos(y) \\implies u_x(z, 0) = e^z \\cos(0) = e^z
u_y(x, y) = -e^x \\sin(y) \\implies u_y(z, 0) = -e^z \\sin(0) = 0
f'(z) = e^z - \\mathrm{i}(0) = e^z
Integrate with respect to z: f(z) = e^z + C.
Apply condition: f(0) = 1 \\implies e^0 + C = 1 \\implies C = 0.
f(z) = e^z.
</think>
\\boxed{e^z}"""},
    {"role": "user", "content": "A test has $H_0: \\mu = 50$ vs $H_a: \\mu < 50$. Sample size $n=16$, sample mean $48$, standard deviation $4$. Find: A. Test statistic. B. Rejection region for $\\alpha=0.05$."},
    {"role": "assistant", "content": r"""<think>
t = \\frac{\\bar{x} - \\mu_0}{s / \\sqrt{n}}
t = \\frac{48 - 50}{4 / \\sqrt{16}} = \\frac{-2}{1} = -2.
df = 15. Left-tailed test, \\alpha = 0.05. Critical t_{0.05, 15} = -1.753.
Rejection region: (-\\infty, -1.753).
</think>
\\boxed{-2, (-\\infty, -1.753)}"""}
]

few_shot_history_mcq = [
    {"role": "system", "content": SYSTEM_PROMPT_MCQ},
    {"role": "user", "content": "Evaluate the contour integral $\\oint_{|z|=2} \\frac{e^z}{z(z-1)^2} \\mathrm{d}z$ where the contour is oriented counterclockwise.\n\nOptions:\nA. $$2\\pi i e$$\nB. $$2\\pi i (e - 1)$$\nC. $$2\\pi i$$\nD. $$0$$\nE. $$-2\\pi i$$\nF. $$4\\pi i e$$\nG. $$\\pi i$$\nH. $$\\pi i e$$\nI. $$2\\pi i (e+1)$$\nJ. $$-2\\pi i e$$"},
    {"role": "assistant", "content": r"""<think>
Identify singularities: simple pole at z = 0, and a pole of order 2 at z = 1.
Both poles lie inside the contour |z| = 2.
Residue at z = 0: \\lim_{z \\to 0} z \\frac{e^z}{z(z-1)^2} = \\frac{e^0}{(-1)^2} = 1.
Residue at z = 1: \\lim_{z \\to 1} \\frac{d}{dz} \\left[ (z-1)^2 \\frac{e^z}{z(z-1)^2} \\right] = \\lim_{z \\to 1} \\frac{d}{dz} \\left( \\frac{e^z}{z} \\right).
Derivative using quotient rule: \\frac{z e^z - e^z}{z^2}.
Evaluate at z = 1: \\frac{1 \\cdot e^1 - e^1}{1^2} = 0.
Sum of residues: 1 + 0 = 1.
By Cauchy's Residue Theorem: \\oint = 2\\pi i \\sum \\text{Res} = 2\\pi i (1) = 2\\pi i.
Matches option C.
</think>
\\boxed{C}"""},
    {"role": "user", "content": "Find the family of orthogonal trajectories to the family of parabolas $y = c x^2$.\n\nOptions:\nA. $$x^2 - 2y^2 = C$$\nB. $$x^2 + y^2 = C$$\nC. $$x^2 + 2y^2 = C$$\nD. $$2x^2 + y^2 = C$$\nE. $$x^2 + 4y^2 = C$$\nF. $$x^2 - y^2 = C$$\nG. $$2x^2 - y^2 = C$$\nH. $$x^2 + y = C$$\nI. $$x + y^2 = C$$\nJ. $$x^2 y = C$$"},
    {"role": "assistant", "content": r"""<think>
Differentiate y = cx^2 with respect to x: y' = 2cx.
Eliminate the parameter c using the original equation c = y/x^2.
Substitute c into the derivative: y' = 2(y/x^2)x = \\frac{2y}{x}.
The differential equation of the original family is y' = \\frac{2y}{x}.
For the orthogonal trajectories, replace y' with -1/y'.
\\frac{-1}{y'} = \\frac{2y}{x} \\implies y' = \\frac{-x}{2y}.
Separate variables: 2y dy = -x dx.
Integrate both sides: \\int 2y dy = \\int -x dx \\implies y^2 = -\\frac{x^2}{2} + K.
Multiply by 2: 2y^2 = -x^2 + 2K \\implies x^2 + 2y^2 = C.
Matches option C.
</think>
\\boxed{C}"""},
    {"role": "user", "content": "We now define an algorithm: The definition of a(n) is the coordination sequence for a specific lattice. Given the input x_list: [1, 2, 3], determine the corresponding output sequence y_list.\n\nOptions:\nA. [4, 8, 12]\nB. [4, 9, 14]\nC. [6, 12, 18]\nD. [6, 18, 38]\nE. [8, 16, 24]"},
    {"role": "assistant", "content": r"""<think>
The problem asks to match the specific algorithm's output sequence to the given options for x = 1, 2, 3.
By analyzing the sequence generation properties for the requested lattice, the values grow sequentially as 4, 8, 12.
Comparing this derived array [4, 8, 12] to the given options.
Matches option A.
</think>
\\boxed{A}"""}
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

# -----------------------------
# Helper functions (local runner)
# -----------------------------
def loadPrivateQuestions(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def extractFinalAnswer(text: str) -> Optional[str]:
    """Extract the content of the last \boxed{...} if present.
    Fallbacks to the last inline $...$ group or the last standalone number.
    """
    if not text:
        return None
    # boxed
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    # inline $...$
    m = re.findall(r"\$(.*?)\$", text, flags=re.DOTALL)
    if m:
        return m[-1].strip()
    # last number-ish token
    m = re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", text)
    if m:
        return m[-1]
    return None


def majorityVote(answers: List[Optional[str]]) -> Dict[str, Any]:
    """Return majority answer info.
    answers: list of extracted answers (may contain None)
    """
    cleaned = [a if a is not None else "" for a in answers]
    counts: Dict[str, int] = {}
    for a in cleaned:
        counts[a] = counts.get(a, 0) + 1
    # ignore empty string key when considering majority
    candidates = [(k, v) for k, v in counts.items() if k != ""]
    if not candidates:
        return {"majority": None, "status": "parse_failed"}
    candidates.sort(key=lambda x: x[1], reverse=True)
    best, best_count = candidates[0]
    if best_count >= 2:
        return {"majority": best, "count": best_count, "status": "majority"}
    else:
        return {"majority": None, "count": best_count, "status": "no_majority"}


def compare_predictions(pred1: str, pred2: str, judger_obj, timeout_sec: int = 4) -> bool:
    """Compare two extracted-answer strings using `Judger` equivalence checking.
    Follows the approach in `run_task.py`.
    """
    list1 = judger_obj.split_by_comma(pred1)
    list2 = judger_obj.split_by_comma(pred2)

    if len(list1) != len(list2):
        return False

    for item1, item2 in zip(list1, list2):
        norm1 = judger_obj.norm_ans_str(item1)
        norm2 = judger_obj.norm_ans_str(item2)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        try:
            is_match = judger_obj.is_equal(norm1, norm2)
        except TimeoutException:
            is_match = False
        except Exception:
            is_match = False
        finally:
            signal.alarm(0)

        if not is_match:
            return False

    return True


def get_mathematical_majority_vote(responses: List[str]) -> Dict[str, Any]:
    """Group responses by mathematical equivalence using `Judger` and return
    the representative (shortest) winning trace and status.
    Returns dict with keys: representative_trace, status, majority_answer, group_count
    """
    from judger import Judger

    jud = Judger(strict_extract=False)

    extracted_data = []
    for resp in responses:
        ans = jud.extract_ans(resp)
        if ans:
            extracted_data.append({"raw_extracted": ans, "raw_response": resp})

    if not extracted_data:
        return {"representative_trace": responses[0] if responses else "", "status": "parse_failed", "majority_answer": None, "group_count": 0}

    equivalence_groups: List[Dict[str, Any]] = []

    for item in extracted_data:
        ans = item["raw_extracted"]
        raw_text = item["raw_response"]
        matched = False

        for group in equivalence_groups:
            if compare_predictions(ans, group["representative"], jud):
                group["count"] += 1
                group["raw_responses"].append(raw_text)
                matched = True
                break

        if not matched:
            equivalence_groups.append({"representative": ans, "count": 1, "raw_responses": [raw_text]})

    # choose winner: most counts, tie-breaker shortest raw response
    best_group = max(equivalence_groups, key=lambda x: (x["count"], -len(min(x["raw_responses"], key=len))))
    representative_trace = min(best_group["raw_responses"], key=len)
    status = "majority" if best_group["count"] >= 2 else "no_majority"
    return {"representative_trace": representative_trace, "status": status, "majority_answer": best_group["representative"], "group_count": best_group["count"]}


def alreadyCompletedIds(output_path: str) -> set:
    s = set()
    p = Path(output_path)
    if not p.exists():
        return s
    valid_statuses = {"majority", "no_majority", "parse_failed", "prompt_too_long"}
    with open(p, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                rec_id = rec.get("id")
                if rec_id is None:
                    continue
                if rec.get("status") in valid_statuses:
                    s.add(rec_id)
            except Exception:
                continue
    return s


def writeJsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Modal worker (remote) - runs inside GPU container
# -----------------------------
app = modal.App(MODAL_STUB_NAME)


# Build the Modal function image using Python 3.11 and install compiler tools.
# This avoids the raw NVIDIA Docker image path that was still running Python 3.9.
try:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("gcc", "g++", "build-essential")
        .pip_install(VLLM_IMAGE_PACKAGES)
    )
except Exception:
    # Fallback to the same Python 3.11 image if the first path fails
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("gcc", "g++", "make", "cmake")
        .pip_install(VLLM_IMAGE_PACKAGES)
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=6 * 60 * 60,  # 6 hours per invocation (tunable)
    secrets=[modal.Secret.from_name("hf_token")],
)
def worker_run(batch_items: List[Dict[str, Any]]):
    """Remote worker: accept a small batch of items and return generated traces.
    Each item should be a dict with keys: id, question, options (optional)
    Returns a list of dicts with trace results and optional error info.
    """
    import os
    import sys
    import shutil

    disable_optional_vllm_kernels()
    from vllm import SamplingParams

    print("REMOTE PYTHON VERSION:", sys.version)
    print("REMOTE PYTHON EXECUTABLE:", sys.executable)
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print("nvcc:", shutil.which("nvcc"))
    print("gpu:", GPU_TYPE)

    # Ensure HF token is available inside the container. Modal secrets: HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    tokenizer, llm = get_worker_model_and_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        n=N_GEN,
    )

    prompts = []
    batch_plan: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for item in batch_items:
        question = item["question"]
        options = item.get("options")
        messages = build_messages(question, options)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompt_len = len(tokenizer.encode(prompt_text))
        print("prompt_tokens:", prompt_len)
        print("max_model_len:", MAX_MODEL_LEN)
        print("max_tokens:", MAX_TOKENS)
        print("n_gen:", N_GEN)

        if prompt_len + MAX_TOKENS > MAX_MODEL_LEN:
            results.append({
                "id": item.get("id"),
                "traces": [],
                "status": "prompt_too_long",
                "error": "prompt_too_long",
                "prompt_tokens": prompt_len,
                "max_model_len": MAX_MODEL_LEN,
                "max_tokens": MAX_TOKENS,
            })
            continue

        batch_plan.append({"id": item.get("id"), "prompt": prompt_text})

    if batch_plan:
        outs = llm.generate([x["prompt"] for x in batch_plan], sampling_params=sampling_params)
        for item, out in zip(batch_plan, outs):
            generated_texts = [comp.text.strip() for comp in out.outputs]
            results.append({"id": item["id"], "traces": generated_texts})

    return results


# -----------------------------
# Local orchestration / entrypoint
# -----------------------------
_worker_tokenizer = None
_worker_llm = None


def get_worker_model_and_tokenizer():
    global _worker_tokenizer, _worker_llm
    if _worker_llm is None or _worker_tokenizer is None:
        disable_optional_vllm_kernels()
        import transformers
        import tokenizers
        import vllm
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        print("vllm version:", vllm.__version__)
        print("transformers version:", transformers.__version__)
        print("tokenizers version:", tokenizers.__version__)

        llm = vllm.LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            quantization=None,
            load_format="auto",
            enable_prefix_caching=True,
            gpu_memory_utilization=0.95,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            max_num_seqs=MAX_NUM_SEQS,
            enforce_eager=True,
        )

        _worker_tokenizer = tokenizer
        _worker_llm = llm

    return _worker_tokenizer, _worker_llm


def generateTraceBatch(local_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call the Modal worker for a small batch and return remote results."""
    if hasattr(worker_run, "remote"):
        return worker_run.remote(local_batch)
    if hasattr(worker_run, "call"):
        return worker_run.call(local_batch)
    return worker_run(local_batch)


def run_local(dry_run: bool = True, resume: bool = True, max_items: Optional[int] = None):
    items = loadPrivateQuestions(DATA_PATH)
    if max_items is not None:
        items = items[:max_items]
    completed = alreadyCompletedIds(OUTPUT_FILE) if resume else set()
    print(f"Loaded {len(items)} items. Already completed: {len(completed)}")

    def process_loop():
        batch: List[Dict[str, Any]] = []
        to_write: List[Dict[str, Any]] = []

        for idx, item in enumerate(items):
            qid = item.get("id")
            if qid in completed:
                continue

            batch.append({
                "id": qid,
                "question": item.get("question"),
                "options": item.get("options"),
            })

            if len(batch) >= BATCH_SIZE or idx == len(items) - 1:
                print(f"Processing batch of {len(batch)}... (dry_run={dry_run})")

                if dry_run:
                    for b in batch:
                        to_write.append({
                            "id": b.get("id"),
                            "question": b.get("question"),
                            "status": "dry_run",
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                else:
                    try:
                        remote_results = generateTraceBatch(batch)
                    except Exception as e:
                        traceback_str = traceback.format_exc()
                        print(f"Remote generation failed: {e}")
                        print(traceback_str)
                        for b in batch:
                            rec = {
                                "id": b.get("id"),
                                "question": b.get("question"),
                                "original_answer": None,
                                "generated_traces": [],
                                "extracted_answers": [],
                                "majority_answer": None,
                                "selected_trace": None,
                                "status": "error",
                                "error": str(e),
                                "traceback": traceback_str,
                                "model": MODEL_ID,
                                "system_prompt": SYSTEM_PROMPT_MCQ if b.get("options") else SYSTEM_PROMPT_MATH,
                                "sampling": {"n": N_GEN, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS},
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            to_write.append(rec)
                    else:
                        from judger import Judger
                        jud = Judger(strict_extract=False)

                        for r in remote_results:
                            uid = r.get("id")
                            source_item = next((it for it in items if it.get("id") == uid), {})

                            if r.get("status") == "prompt_too_long":
                                to_write.append({
                                    "id": uid,
                                    "question": source_item.get("question"),
                                    "original_answer": source_item.get("answer"),
                                    "generated_traces": [],
                                    "extracted_answers": [],
                                    "majority_answer": None,
                                    "selected_trace": None,
                                    "status": "prompt_too_long",
                                    "error": r.get("error"),
                                    "prompt_tokens": r.get("prompt_tokens"),
                                    "max_model_len": r.get("max_model_len"),
                                    "max_tokens": r.get("max_tokens"),
                                    "model": MODEL_ID,
                                    "system_prompt": SYSTEM_PROMPT_MCQ if source_item.get("options") else SYSTEM_PROMPT_MATH,
                                    "sampling": {"n": N_GEN, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS},
                                    "timestamp": datetime.utcnow().isoformat(),
                                })
                                continue

                            if r.get("error"):
                                to_write.append({
                                    "id": uid,
                                    "question": source_item.get("question"),
                                    "original_answer": source_item.get("answer"),
                                    "generated_traces": [],
                                    "extracted_answers": [],
                                    "majority_answer": None,
                                    "selected_trace": None,
                                    "status": "error",
                                    "error": r.get("error"),
                                    "traceback": r.get("traceback"),
                                    "model": MODEL_ID,
                                    "system_prompt": SYSTEM_PROMPT_MCQ if source_item.get("options") else SYSTEM_PROMPT_MATH,
                                    "sampling": {"n": N_GEN, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS},
                                    "timestamp": datetime.utcnow().isoformat(),
                                })
                                continue

                            traces = r.get("traces", [])
                            extracted = [jud.extract_ans(t) for t in traces]

                            mv = get_mathematical_majority_vote(traces)
                            status = mv.get("status")
                            majority_answer = mv.get("majority_answer")
                            representative_trace = mv.get("representative_trace")

                            selected_trace = representative_trace if status == "majority" else None

                            to_write.append({
                                "id": uid,
                                "question": source_item.get("question"),
                                "original_answer": source_item.get("answer"),
                                "generated_traces": traces,
                                "extracted_answers": extracted,
                                "majority_answer": majority_answer,
                                "selected_trace": selected_trace,
                                "status": status,
                                "model": MODEL_ID,
                                "system_prompt": SYSTEM_PROMPT_MCQ if source_item.get("options") else SYSTEM_PROMPT_MATH,
                                "sampling": {"n": N_GEN, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS},
                                "timestamp": datetime.utcnow().isoformat(),
                            })

                if to_write:
                    writeJsonl(to_write, OUTPUT_FILE)
                    print(f"Wrote {len(to_write)} records to {OUTPUT_FILE}")
                    to_write = []

                batch = []

    if dry_run:
        process_loop()
    else:
        with app.run():
            process_loop()

    print("Run complete.")


def create_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true", help="Do not call Modal; just simulate and check batching logic.")
    p.add_argument("--resume", action="store_true", default=True, help="Skip already-completed ids in OUTPUT_FILE.")
    p.add_argument("--n_gen", type=int, default=N_GEN, help="Number of generations per question.")
    p.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling.")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Maximum tokens to generate.")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for Modal calls.")
    p.add_argument("--max_num_seqs", type=int, default=MAX_NUM_SEQS, help="Modal max_num_seqs for vLLM.")
    p.add_argument("--max_items", type=int, default=None, help="Only process the first N items for a smoke test.")
    return p


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    N_GEN = args.n_gen
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size
    MAX_NUM_SEQS = args.max_num_seqs
    run_local(dry_run=args.dry_run, resume=args.resume, max_items=args.max_items)
