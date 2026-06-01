"""End-to-end Modal inference pipeline for the CSE 151B math task.

This script combines the Modal/vLLM execution shape from
``distillation/run_distillation.py`` with the routed prompt construction in
``prompt_strategy.py``. The main remote entrypoint is ``run_inference()``.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Optional

import modal


DEFAULT_MODEL_ID = "hzia360/qwen3-4b-sft-merged2"
DEFAULT_PUBLIC_DATA = "data/public.jsonl"
DEFAULT_PRIVATE_DATA = "data/private.jsonl"

app = modal.App("cse151b-routed-inference")
vol = modal.Volume.from_name("inference-results", create_if_missing=True)


def download_default_model() -> None:
    """Bake the default model into the Modal image when credentials allow it."""

    from huggingface_hub import snapshot_download

    snapshot_download(DEFAULT_MODEL_ID)


image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel")
    .env(
        {
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .pip_install(
        "vllm",
        "transformers",
        "tqdm",
        "numpy",
        "antlr4-python3-runtime==4.11.1",
        "sympy",
        "huggingface_hub",
        "hf_transfer",
    )
    .run_function(
        download_default_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("judger.py", remote_path="/root/judger.py")
    .add_local_file("prompt_strategy.py", remote_path="/root/prompt_strategy.py")
    .add_local_dir("data", remote_path="/root/data")
)


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame) -> None:
    raise TimeoutException("Sympy evaluation took too long.")


def extract_letter(text: str) -> str:
    match = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_item(item: dict, response: str) -> dict:
    """Score one public-data item in a separate process."""

    from judger import Judger

    is_mcq = bool(item.get("options"))
    gold = item["answer"]

    if is_mcq:
        correct = extract_letter(response) == str(gold).strip().upper()
    else:
        judger = Judger(strict_extract=False)
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[] for _ in gold_list],
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


def majority_vote(generated_texts: list[str]) -> str:
    """Group candidate answers by mathematical equivalence and return a winner."""

    from judger import Judger

    judger = Judger(strict_extract=False)

    def compare_predictions(pred1: str, pred2: str, timeout_sec: int = 4) -> bool:
        if pred1.replace(" ", "") == pred2.replace(" ", ""):
            return True

        list1 = judger.split_by_comma(pred1)
        list2 = judger.split_by_comma(pred2)
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            norm1 = judger.norm_ans_str(item1)
            norm2 = judger.norm_ans_str(item2)

            signal.signal(signal.SIGALRM, _timeout_handler)
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

    extracted = []
    for response in generated_texts:
        answer = judger.extract_ans(response)
        if answer:
            extracted.append({"answer": answer, "response": response})

    if not extracted:
        return generated_texts[0] if generated_texts else ""

    groups: list[dict] = []
    for item in extracted:
        matched = False
        for group in groups:
            if compare_predictions(item["answer"], group["representative"]):
                group["count"] += 1
                group["responses"].append(item["response"])
                matched = True
                break

        if not matched:
            groups.append(
                {
                    "representative": item["answer"],
                    "count": 1,
                    "responses": [item["response"]],
                }
            )

    best_group = max(
        groups,
        key=lambda group: (
            group["count"],
            -len(min(group["responses"], key=len)),
        ),
    )
    return min(best_group["responses"], key=len)


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_submission_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "response": row["response"]})


def _summarize(results: list[dict]) -> str:
    def accuracy_line(label: str, subset: list[dict]) -> str:
        total = len(subset)
        correct = sum(row.get("correct", False) for row in subset)
        pct = correct / total * 100 if total else 0.0
        return f"{label:10s}: {correct:4d} / {total:4d} ({pct:6.2f}%)"

    mcq = [row for row in results if row["is_mcq"]]
    free = [row for row in results if not row["is_mcq"]]
    return "\n".join(
        [
            accuracy_line("MCQ", mcq),
            accuracy_line("Free-form", free),
            accuracy_line("Overall", results),
        ]
    )


def _normalize_config(config: dict) -> dict:
    normalized = dict(config)
    evaluation = bool(normalized.get("eval", False))

    if not normalized.get("data_path"):
        normalized["data_path"] = DEFAULT_PUBLIC_DATA if evaluation else DEFAULT_PRIVATE_DATA
    if not normalized.get("output_path"):
        normalized["output_path"] = (
            "/results/routed_public_eval.jsonl"
            if evaluation
            else "/results/routed_private_inference.jsonl"
        )
    normalized.setdefault("model_id", DEFAULT_MODEL_ID)
    normalized.setdefault("num_samples", None)
    normalized.setdefault("num_outputs", 1)
    normalized.setdefault("temperature", 0.2)
    normalized.setdefault("top_p", 0.95)
    normalized.setdefault("max_tokens", 8192)
    normalized.setdefault("max_num_seqs", 64)
    normalized.setdefault("tensor_parallel_size", 2)
    normalized.setdefault("prompt_mode", "routed")
    normalized.setdefault("example_source", "curated")
    normalized.setdefault("shots", 1)
    return normalized


@app.function(
    image=image,
    gpu="H200:2",
    timeout=86400,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_inference(config: dict) -> dict:
    """Run routed prompt construction, vLLM generation, voting, and output writing."""

    os.chdir("/root")
    sys.path.insert(0, "/root")

    from prompt_strategy import (
        build_prompt,
        classify_math_topic,
        select_curated_examples,
        select_few_shot_examples,
    )
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args = _normalize_config(config)
    evaluation = bool(args["eval"])

    data = _load_jsonl(args["data_path"])
    if args["num_samples"] is not None:
        data = data[: int(args["num_samples"])]

    if not data:
        raise ValueError(f"No records loaded from {args['data_path']}")

    n_mcq = sum(bool(item.get("options")) for item in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    tokenizer = AutoTokenizer.from_pretrained(args["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print("Topic mix:")
    topic_counts: dict[str, int] = {}
    for record in prompt_records:
        topic_counts[record["topic"]] = topic_counts.get(record["topic"], 0) + 1
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]

    llm = LLM(
        model=args["model_id"],
        dtype="bfloat16",
        tensor_parallel_size=int(args["tensor_parallel_size"]),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_model_len=int(args["max_tokens"]) + 4096,
        trust_remote_code=True,
        max_num_seqs=int(args["max_num_seqs"]),
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(
        max_tokens=int(args["max_tokens"]),
        temperature=float(args["temperature"]),
        top_p=float(args["top_p"]),
        top_k=20,
        min_p=0.0,
        n=int(args["num_outputs"]),
        presence_penalty=0.0,
        repetition_penalty=1.05,
    )

    print(f"Generating {len(prompts)} prompts with {args['model_id']}")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    all_generated_texts = [[choice.text.strip() for choice in output.outputs] for output in outputs]

    if int(args["num_outputs"]) > 1:
        print("Running parallel mathematical majority voting")
        responses = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            iterator = executor.map(majority_vote, all_generated_texts)
            for response in tqdm(iterator, total=len(all_generated_texts), desc="Voting"):
                responses.append(response)
    else:
        responses = [texts[0] if texts else "" for texts in all_generated_texts]

    results = []
    if evaluation:
        print("Scoring public evaluation data")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            iterator = executor.map(score_item, data, responses)
            for scored, prompt_record in tqdm(
                zip(iterator, prompt_records),
                total=len(data),
                desc="Scoring",
            ):
                scored["topic"] = prompt_record["topic"]
                scored["few_shot_ids"] = prompt_record["few_shot_ids"]
                results.append(scored)
        print(_summarize(results))
    else:
        for item, response, prompt_record in zip(data, responses, prompt_records):
            results.append(
                {
                    "id": item.get("id"),
                    "is_mcq": bool(item.get("options")),
                    "topic": prompt_record["topic"],
                    "few_shot_ids": prompt_record["few_shot_ids"],
                    "response": response,
                }
            )

    out_path = Path(args["output_path"])
    _write_jsonl(out_path, results)
    print(f"Saved {len(results)} JSONL records to {out_path}")

    csv_path: Optional[Path] = None
    if not evaluation:
        csv_path = out_path.with_suffix(".csv")
        _write_submission_csv(csv_path, results)
        print(f"Saved submission CSV to {csv_path}")

    vol.commit()
    return {
        "output_path": str(out_path),
        "csv_path": str(csv_path) if csv_path else None,
        "records": len(results),
        "evaluation": evaluation,
    }


@app.local_entrypoint()
def main(
    eval: bool = False,
    data_path: str = "",
    output_path: str = "",
    model_id: str = DEFAULT_MODEL_ID,
    num_samples: int = 0,
    num_outputs: int = 15,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    max_num_seqs: int = 64,
    tensor_parallel_size: int = 2,
    prompt_mode: str = "routed",
    example_source: str = "curated",
    shots: int = 1,
    detach: bool = False,
) -> None:
    """Modal CLI wrapper.

    Example:
        modal run run_inference.py --eval --num-samples 10
        modal run run_inference.py --output-path /results/private_run.jsonl
    """

    if prompt_mode not in {"baseline", "routed"}:
        raise ValueError("--prompt-mode must be baseline or routed")
    if example_source not in {"curated", "dataset", "mixed", "none"}:
        raise ValueError("--example-source must be curated, dataset, mixed, or none")

    config = {
        "eval": eval,
        "data_path": data_path or None,
        "output_path": output_path or None,
        "model_id": model_id,
        "num_samples": num_samples if num_samples > 0 else None,
        "num_outputs": num_outputs,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": tensor_parallel_size,
        "prompt_mode": prompt_mode,
        "example_source": example_source,
        "shots": shots,
    }

    if detach:
        call = run_inference.spawn(config)
        print(f"Started detached Modal inference call: {call.object_id}")
    else:
        result = run_inference.remote(config)
        print(json.dumps(result, indent=2))
