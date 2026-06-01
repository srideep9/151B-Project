"""Reranking prompt helpers for inference-time candidate verification."""

from __future__ import annotations

import re
from typing import Optional


RERANK_SYSTEM_PROMPT = (
    "You are a strict mathematical answer verifier. You will receive one math "
    "problem and several candidate solution traces from another model. Check the "
    "reasoning, arithmetic, final boxed answer, formatting, answer order, and any "
    "multiple-choice constraints. Choose the single candidate most likely to be "
    "correct. Do not prefer a candidate just because it is longer. Output exactly "
    "one line in the form CHOICE: <number>."
)


def format_options(options: Optional[list[str]]) -> str:
    if not options:
        return ""
    labels = [chr(65 + idx) for idx in range(len(options))]
    return "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))


def extract_boxed_answer(text: str) -> str:
    """Return the final boxed answer using a lightweight balanced-brace scan."""

    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""

    idx = start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        idx += 1
    return ""


def compact_trace(text: str, max_chars: int = 1600) -> str:
    """Keep enough of a trace to audit it without overflowing rerank context."""

    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= max_chars:
        return text

    tail_chars = max(500, max_chars // 2)
    head_chars = max_chars - tail_chars
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()
    return f"{head}\n\n[... trace truncated ...]\n\n{tail}"


def build_rerank_messages(
    question: str,
    options: Optional[list[str]],
    candidates: list[str],
    *,
    max_trace_chars: int = 1600,
) -> list[dict[str, str]]:
    """Build chat messages asking the model to choose the best candidate trace."""

    options_text = format_options(options)
    problem = f"{question}\n\nOptions:\n{options_text}" if options_text else question
    candidate_blocks = []
    for idx, candidate in enumerate(candidates, start=1):
        answer = extract_boxed_answer(candidate) or "[no boxed answer found]"
        trace = compact_trace(candidate, max_chars=max_trace_chars)
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate {idx}",
                    f"Final boxed answer: {answer}",
                    "Trace:",
                    trace,
                ]
            )
        )

    candidates_text = "\n\n---\n\n".join(candidate_blocks)
    user_prompt = (
        "Problem:\n"
        f"{problem}\n\n"
        "Candidate solution traces:\n\n"
        f"{candidates_text}\n\n"
        "Choose the candidate with the correct final answer. Output only "
        "CHOICE: <number>."
    )
    return [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_rerank_choice(text: str, num_candidates: int) -> Optional[int]:
    """Parse a 0-based candidate index from verifier output."""

    patterns = (
        r"\bCHOICE\s*:\s*(\d+)\b",
        r"\b(?:answer|selection)\s*:\s*(\d+)\b",
        r'"choice"\s*:\s*(\d+)\b',
        r"\bCandidate\s+(\d+)\b",
        r"^\s*(\d+)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        choice = int(match.group(1))
        if 1 <= choice <= num_candidates:
            return choice - 1
    return None
