"""Prompt construction helpers for quick math-reasoning experiments.

This module is intentionally lightweight: it does not load a model, and it can
be imported from a notebook, a script, or a future training pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a "
    "single \\boxed{}, e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


@dataclass(frozen=True)
class MathTopic:
    name: str
    hint: str
    keywords: tuple[str, ...]


TOPICS: tuple[MathTopic, ...] = (
    MathTopic(
        name="geometry",
        hint=(
            "Use geometric relationships before algebra: draw or imagine a diagram, "
            "track givens, use angle/triangle/circle/area formulas, and check units."
        ),
        keywords=(
            "triangle", "circle", "radius", "diameter", "angle", "perpendicular",
            "parallel", "polygon", "area", "volume", "surface area", "similar",
            "congruent", "hypotenuse", "rectangle", "square", "trapezoid",
        ),
    ),
    MathTopic(
        name="complex_analysis",
        hint=(
            "Use complex-variable structure. For analytic functions, apply the "
            "Cauchy-Riemann equations, harmonic conjugates, and complex algebra; for "
            "contour or residue problems, identify singularities and relevant theorems."
        ),
        keywords=(
            "analytic function", "complex function", "cauchy-riemann",
            "cauchy riemann", "harmonic conjugate", "residue", "contour",
            "singularity", "singularities", "complex plane", "\\mathrm{i}",
            "imaginary part", "real part", "f(z)", "u(x, y)", "u +",
        ),
    ),
    MathTopic(
        name="derivatives",
        hint=(
            "Differentiate carefully using product, quotient, chain, implicit, or "
            "logarithmic differentiation as needed, then simplify the final expression."
        ),
        keywords=(
            "derivative", "differentiate", "d/d", "\\frac{d", "slope of the tangent",
            "tangent line", "rate of change", "critical point", "increasing",
            "decreasing", "interval on which", "y^{", "y^(",
        ),
    ),
    MathTopic(
        name="trigonometry",
        hint=(
            "Use trig identities and right-triangle definitions. Keep track of radians "
            "versus degrees, special angles, and quadrant signs."
        ),
        keywords=(
            "sin", "cos", "tan", "cot", "sec", "csc", "trigonometric",
            "radian", "degree", "\\sin", "\\cos", "\\tan", "unit circle",
        ),
    ),
    MathTopic(
        name="integrals",
        hint=(
            "Identify the integrand structure first. Consider substitution, parts, "
            "symmetry, bounds, and constants of integration when relevant."
        ),
        keywords=(
            "integral", "integrate", "antiderivative", "area under", "\\int",
            "int_", "int^", "definite integral", "indefinite integral",
        ),
    ),
    MathTopic(
        name="gradients",
        hint=(
            "Treat the expression as multivariable. Compute partial derivatives, form "
            "the gradient vector, and evaluate it at the requested point if given."
        ),
        keywords=(
            "gradient", "partial derivative", "partials", "\\nabla", "jacobian",
            "multivariable", "directional derivative",
        ),
    ),
    MathTopic(
        name="differential_equations",
        hint=(
            "Model the changing quantity with a differential equation. For growth, "
            "decay, half-life, heating, cooling, or Newton's law problems, identify "
            "the ambient/equilibrium value, initial condition, and time constant."
        ),
        keywords=(
            "differential equation", "half-life", "half life", "population growth",
            "population grows", "population decay", "exponential growth",
            "exponential decay", "cooling", "heating", "newton's law",
            "newtons law", "newton law", "ambient", "room temperature",
            "rate of change is proportional", "temperature after",
        ),
    ),
    MathTopic(
        name="algebra",
        hint=(
            "Translate the statement into equations or arithmetic steps, simplify "
            "expressions, factor or expand when useful, and check candidate solutions "
            "against constraints. For simple word problems, identify the quantities, "
            "operations, and requested unknown before solving."
        ),
        keywords=(
            "solve", "equation", "system", "polynomial", "quadratic", "linear",
            "factor", "expand", "simplify", "expression", "variable", "roots",
            "inequality", "word problem", "total", "altogether", "how many",
            "how much", "cost", "price", "average", "mean", "fraction", "reduce",
            "percent", "percentage", "ratio", "proportion", "rate", "distance",
            "speed", "work", "mixture", "age", "older", "younger", "binary numbers",
            "binary", "simplified formula", "function below", "u(x)", "v(x)",
            "add and subtract", "add and subtract mentally", "algorithm", "x_list",
            "y_list", "output sequence",
        ),
    ),
    MathTopic(
        name="sequences_series",
        hint=(
            "Identify the sequence pattern, first term, common difference or ratio, "
            "and number of terms. Use the finite arithmetic or geometric sum formula "
            "when appropriate."
        ),
        keywords=(
            "sequence", "series", "sum of the first", "positive even",
            "arithmetic", "geometric", "common difference", "common ratio",
            "terms", "nth term", "\\dots", "\\cdots", "lfloor", "floor",
            "[ \\alpha", "\\alpha+", "remainder",
        ),
    ),
    MathTopic(
        name="statistics",
        hint=(
            "Organize the data or distribution information first. Compute requested "
            "statistics such as mean, variance, standard deviation, percentiles, or "
            "order-statistic probabilities with careful notation and rounding."
        ),
        keywords=(
            "standard deviation", "variance", "percentile", "order statistics",
            "independent observations", "continuous-type distribution", "mean",
            "median", "mode", "quartile", "data set", "sample", "population",
            "confidence interval", "hypothesis test", "p-value", "z-score",
            "normal distribution", "distribution", "hypothesis", "null hypothesis",
            "level of significance", "significance", "h_0", "h1", "h_1",
        ),
    ),
    MathTopic(
        name="probability",
        hint=(
            "Define the sample space and favorable outcomes. Use complements, "
            "conditioning, independence, or counting formulas when appropriate. "
            "For combinatorics-style questions, decide whether order matters, whether "
            "repetition is allowed, and whether inclusion-exclusion is needed."
        ),
        keywords=(
            "probability", "random", "expected", "expectation", "dice", "coin",
            "cards", "at least", "given that", "count",
            "ways", "arrangements", "permutations", "combinations", "choose",
            "binomial", "subsets", "how many ways", "number of ways", "arranged",
            "without replacement", "with replacement",
        ),
    ),
)


DEFAULT_TOPIC = MathTopic(
    name="general",
    hint=(
        "Choose the most direct mathematical representation, solve in small verified "
        "steps, and sanity-check the final boxed answer."
    ),
    keywords=(),
)


def classify_math_topic(question: str) -> MathTopic:
    """Classify a math problem with transparent keyword rules.

    The rules are deliberately simple so local prompt experiments are fast and
    inspectable. A learned or model-based classifier can replace this function
    later without changing the prompt-building interface.
    """

    text = question.lower()
    scores: list[tuple[int, int, MathTopic]] = []
    for order, topic in enumerate(TOPICS):
        score = sum(1 for keyword in topic.keywords if keyword_matches(text, keyword))
        if score:
            scores.append((score, -order, topic))
    if not scores:
        return DEFAULT_TOPIC
    return max(scores, key=lambda item: (item[0], item[1]))[2]


def keyword_matches(text: str, keyword: str) -> bool:
    """Return True when a keyword appears as a phrase, not inside another word."""

    keyword = keyword.lower()
    if keyword.startswith("\\") or any(char in keyword for char in "_^{}"):
        return keyword in text
    if re.search(r"[a-z0-9]", keyword):
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    return keyword in text


def format_options(options: Optional[list[str]]) -> str:
    if not options:
        return ""
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))


def format_answer(answer) -> str:
    if isinstance(answer, list):
        return ", ".join(str(part) for part in answer)
    return str(answer)


def format_example(item: dict) -> str:
    options_text = format_options(item.get("options"))
    question = item["question"]
    if options_text:
        question = f"{question}\nOptions:\n{options_text}"
    return f"Problem:\n{question}\nAnswer:\n\\boxed{{{format_answer(item['answer'])}}}"


def select_few_shot_examples(data: list[dict], target: dict, shots: int) -> list[dict]:
    """Pick same-type, same-topic examples from public data when possible."""

    if shots <= 0:
        return []

    target_topic = classify_math_topic(target["question"]).name
    target_is_mcq = bool(target.get("options"))
    target_id = target.get("id")

    def is_candidate(item: dict) -> bool:
        return (
            item.get("id") != target_id
            and "answer" in item
            and bool(item.get("options")) == target_is_mcq
        )

    same_topic = [
        item for item in data
        if is_candidate(item) and classify_math_topic(item["question"]).name == target_topic
    ]
    fallback = [item for item in data if is_candidate(item) and item not in same_topic]
    return (same_topic + fallback)[:shots]


def build_prompt(
    question: str,
    options: Optional[list[str]] = None,
    *,
    topic: Optional[MathTopic] = None,
    few_shot_examples: Optional[list[dict]] = None,
    include_topic_hint: bool = True,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` with optional topic guidance."""

    topic = topic or classify_math_topic(question)
    system = SYSTEM_PROMPT_MCQ if options else SYSTEM_PROMPT_MATH
    if include_topic_hint:
        system = f"{system} Problem type: {topic.name}. {topic.hint}"

    options_text = format_options(options)
    target_question = f"{question}\n\nOptions:\n{options_text}" if options_text else question

    examples = few_shot_examples or []
    if not examples:
        return system, target_question

    example_text = "\n\n".join(format_example(example) for example in examples)
    user = (
        "Use the examples as answer-format guidance, then solve the final problem.\n\n"
        f"{example_text}\n\nFinal problem:\n{target_question}"
    )
    return system, user


def extract_letter(text: str) -> str:
    match = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""
