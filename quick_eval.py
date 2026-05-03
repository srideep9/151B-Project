"""Fast one-shot/few-shot evaluation harness for the CSE 151B math dataset.

This script avoids notebook execution and defaults to a cheap mock backend so
prompt/classification changes can be inspected locally before running a GPU job.
Use ``--backend transformers`` only when you actually want to load a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from prompt_strategy import (
    build_prompt,
    classify_math_topic,
    extract_letter,
    format_answer,
    select_few_shot_examples,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def extract_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return ""


def score_response(judger, item: dict, response: str) -> bool:
    gold = item["answer"]
    if item.get("options"):
        return score_mcq(response, str(gold))

    if judger is None:
        return extract_boxed(response) == format_answer(gold)

    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def mock_generate(items: Iterable[dict]) -> list[str]:
    """Return gold answers when available, useful for testing scoring plumbing."""

    responses = []
    for item in items:
        answer = format_answer(item.get("answer", ""))
        responses.append(f"Mock reasoning omitted for speed. Final answer: \\boxed{{{answer}}}")
    return responses


def transformers_generate(
    messages_list: list[list[dict[str, str]]],
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for messages in messages_list:
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        prompts.append(prompt)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
    )
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        for output in output_ids
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_judger():
    try:
        from judger import Judger

        return Judger(strict_extract=False)
    except Exception as exc:
        print(f"Warning: Judger unavailable ({exc}). Falling back to exact boxed-answer checks.")
        return None


def summarize(results: list[dict]) -> str:
    mcq = [result for result in results if result["is_mcq"]]
    free = [result for result in results if not result["is_mcq"]]

    def row(name: str, subset: list[dict]) -> str:
        total = len(subset)
        correct = sum(result["correct"] for result in subset)
        accuracy = (correct / total * 100) if total else 0.0
        return f"{name:10s}: {correct:4d} / {total:4d} ({accuracy:6.2f}%)"

    return "\n".join([row("MCQ", mcq), row("Free-form", free), row("Overall", results)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick one-shot/few-shot math reasoning experiments."
    )
    parser.add_argument("--data", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/quick_eval.jsonl"))
    parser.add_argument("--limit", type=int, default=5, help="Number of problems to test.")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the dataset.")
    parser.add_argument("--shots", type=int, default=0, help="Few-shot examples per problem.")
    parser.add_argument(
        "--backend",
        choices=("mock", "transformers"),
        default="mock",
        help="mock is instant and tests prompts/scoring; transformers loads a local model.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Transformers model id; only used with --backend transformers.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print prompts and classifications without generating or scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.data)
    sample = data[args.offset: args.offset + args.limit]
    if not sample:
        raise SystemExit(f"No examples selected from {args.data}")

    messages_list: list[list[dict[str, str]]] = []
    prompt_records: list[dict] = []
    for item in sample:
        topic = classify_math_topic(item["question"])
        examples = select_few_shot_examples(data, item, args.shots)
        system, user = build_prompt(
            item["question"],
            item.get("options"),
            topic=topic,
            few_shot_examples=examples,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>"
        messages_list.append(messages)
        prompt_records.append(
            {
                "id": item.get("id"),
                "topic": topic.name,
                "is_mcq": bool(item.get("options")),
                "few_shot_ids": [example.get("id") for example in examples],
                "prompt": prompt,
            }
        )

    print(f"Selected {len(sample)} problems from {args.data}")
    print("Topic mix:", ", ".join(f"{r['id']}={r['topic']}" for r in prompt_records))

    if args.preview_only:
        for record in prompt_records:
            print("\n" + "=" * 80)
            print(f"id={record['id']} topic={record['topic']} shots={record['few_shot_ids']}")
            print(record["prompt"])
        return

    if args.backend == "mock":
        responses = mock_generate(sample)
    else:
        responses = transformers_generate(
            messages_list,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    judger = load_judger()
    results = []
    for item, prompt_record, response in zip(sample, prompt_records, responses):
        correct = score_response(judger, item, response)
        results.append(
            {
                "id": item.get("id"),
                "topic": prompt_record["topic"],
                "is_mcq": bool(item.get("options")),
                "gold": item.get("answer"),
                "response": response,
                "correct": correct,
            }
        )

    write_jsonl(args.output, results)
    print(summarize(results))
    print(f"Saved {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
