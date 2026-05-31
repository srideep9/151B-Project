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


CURATED_FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "algebra": [
        {
            "description": "Simple arithmetic word problem that translates directly into one or two operations.",
            "question": "Maya buys 3 notebooks for $4 each and a backpack for $28. How much does she spend in total?",
            "approach": "Multiply the number of notebooks by the price per notebook, then add the backpack cost.",
            "answer": "40",
        },
        {
            "description": "Equation-solving word problem with one unknown, such as price, age, tickets, or salary.",
            "question": "A movie theater ticket costs $12. A group pays $84 total. How many tickets did they buy?",
            "approach": "Set up the equation 12x = 84, then solve for x.",
            "answer": "7",
        },
        {
            "description": "Expression simplification or function-composition problem with variables.",
            "question": "If f(x) = 2x + 3 and g(x) = x^2, what is f(g(4))?",
            "options": ["11", "19", "35", "64"],
            "approach": "Evaluate the inside function first: find g(4), then plug that result into f.",
            "answer": "C",
        },
    ],

    "geometry": [
        {
            "description": "Triangle or circle problem requiring a standard formula or theorem.",
            "question": "A right triangle has legs of length 6 and 8. What is the length of the hypotenuse?",
            "options": ["10", "12", "14", "16"],
            "approach": "Use the Pythagorean theorem: a^2 + b^2 = c^2.",
            "answer": "A",
        },
        {
            "description": "Area, volume, or surface-area problem where units and dimensions matter.",
            "question": "A rectangle has length 12 inches and width 5 inches. What is its area?",
            "approach": "Use the rectangle area formula: area = length × width.",
            "answer": "60 square inches",
        },
        {
            "description": "Similar-triangles, angle-chasing, or coordinate-geometry example.",
            "question": "Two similar triangles have side lengths in a ratio of 3:5. If the smaller triangle has a side of length 9, what is the corresponding side length in the larger triangle?",
            "approach": "Use the scale factor from smaller to larger: 5/3.",
            "answer": "15",
        },
    ],

    "trigonometry": [
        {
            "description": "Basic trig equation with all solutions or interval-restricted solutions.",
            "question": "On the interval 0° ≤ x ≤ 360°, which values solve sin(x) = 0?",
            "options": [
                "0° and 180° only",
                "90° and 270° only",
                "0°, 180°, and 360°",
                "45° and 225°",
            ],
            "approach": "Recall where sine is zero on the unit circle.",
            "answer": "C",
        },
        {
            "description": "Identity simplification problem using sine/cosine/tangent identities.",
            "question": "Simplify the expression sin^2(x) + cos^2(x).",
            "approach": "Use the Pythagorean identity for sine and cosine.",
            "answer": "1",
        },
        {
            "description": "Right-triangle or unit-circle problem involving angle units and quadrant signs.",
            "question": "In a right triangle, an angle θ has opposite side 5 and hypotenuse 13. What is sin(θ)?",
            "approach": "Use the definition sin(θ) = opposite / hypotenuse.",
            "answer": "5/13",
        },
    ],

    "integrals": [
        {
            "description": "Substitution integral with a clear inner function and derivative.",
            "question": "Find the indefinite integral: ∫ 2x(x^2 + 1)^3 dx.",
            "approach": "Use u-substitution with u = x^2 + 1.",
            "answer": "(x^2 + 1)^4 / 4 + C",
        },
        {
            "description": "Definite or improper integral where bounds or convergence must be handled carefully.",
            "question": "What is ∫ from 0 to 2 of 3x^2 dx?",
            "options": ["4", "6", "8", "12"],
            "approach": "Find the antiderivative of 3x^2, then evaluate it at the upper and lower bounds.",
            "answer": "C",
        },
        {
            "description": "Rational, partial-fractions, symmetry, or integration-by-parts example.",
            "question": "Find the indefinite integral: ∫ 1/x dx, where x > 0.",
            "approach": "Recall the standard antiderivative of 1/x.",
            "answer": "ln(x) + C",
        },
    ],

    "derivatives": [
        {
            "description": "Product, quotient, or chain-rule derivative with simplification.",
            "question": "Find the derivative of y = (x^2 + 1)^3.",
            "approach": "Use the chain rule: differentiate the outside power, then multiply by the derivative of the inside.",
            "answer": "6x(x^2 + 1)^2",
        },
        {
            "description": "Higher-order derivative or derivative involving trig/exponential terms.",
            "question": "If f(x) = e^x + sin(x), what is f'(x)?",
            "options": [
                "e^x + cos(x)",
                "e^x - cos(x)",
                "x e^(x-1) + cos(x)",
                "e^x + sin(x)",
            ],
            "approach": "Differentiate each term separately.",
            "answer": "A",
        },
        {
            "description": "Increasing/decreasing interval, tangent line, or critical-point example.",
            "question": "Find the equation of the tangent line to y = x^2 at x = 3.",
            "approach": "Find the slope using the derivative, then use point-slope form with the point on the curve.",
            "answer": "y = 6x - 9",
        },
    ],

    "gradients": [
        {
            "description": "Compute a gradient vector for a multivariable scalar function.",
            "question": "For f(x, y) = x^2 + 3y^2, compute ∇f(x, y).",
            "approach": "Take the partial derivative with respect to x, then with respect to y.",
            "answer": "∇f(x, y) = <2x, 6y>",
        },
        {
            "description": "Evaluate a gradient at a point.",
            "question": "If f(x, y) = x^2y + y^2, what is ∇f(2, 3)?",
            "options": ["<12, 10>", "<6, 7>", "<4, 12>", "<18, 8>"],
            "approach": "First compute the gradient symbolically, then substitute x = 2 and y = 3.",
            "answer": "A",
        },
        {
            "description": "Directional derivative, Jacobian, or partial-derivative interpretation example.",
            "question": "For f(x, y) = 4x + 2y, what is the directional derivative in the unit direction <1, 0>?",
            "approach": "Compute the gradient and take its dot product with the given unit direction.",
            "answer": "4",
        },
    ],

    "differential_equations": [
        {
            "description": "Half-life or exponential decay problem requiring a constant from data.",
            "question": "A substance has a half-life of 10 years. If you start with 80 grams, how much remains after 20 years?",
            "approach": "After each half-life, the amount is cut in half. Count how many half-lives have passed.",
            "answer": "20",
        },
        {
            "description": "Population growth/decay problem with initial value and later prediction.",
            "question": "A population starts at 500 and grows by 10% each year. Which expression gives the population after 3 years?",
            "options": [
                "500 + 0.1(3)",
                "500(1.1)^3",
                "500(0.9)^3",
                "500 + 10^3",
            ],
            "approach": "Use exponential growth: initial amount times growth factor raised to the number of years.",
            "answer": "B",
        },
        {
            "description": "Newton heating/cooling problem with ambient temperature and elapsed time.",
            "question": "A hot drink cools according to T(t) = 70 + 90e^(-0.2t), where T is in degrees Fahrenheit and t is in minutes. What is the room temperature?",
            "approach": "In Newton's cooling model, the constant term represents the ambient temperature.",
            "answer": "70",
        },
    ],

    "complex_analysis": [
        {
            "description": "Analytic function from real part u(x,y), using Cauchy-Riemann equations.",
            "question": "Suppose f(z) = u(x, y) + iv(x, y) is analytic and u(x, y) = x^2 - y^2. What is one possible imaginary part v(x, y)?",
            "approach": "Use the Cauchy-Riemann equations to relate partial derivatives of u and v.",
            "answer": "v(x, y) = 2xy + C",
        },
        {
            "description": "Complex algebra problem involving real/imaginary parts or powers of z.",
            "question": "If z = 3 + 4i, what is |z|?",
            "options": ["1", "5", "7", "25"],
            "approach": "Use the modulus formula |a + bi| = sqrt(a^2 + b^2).",
            "answer": "B",
        },
        {
            "description": "Residue, contour, singularity, or harmonic-conjugate example.",
            "question": "What type of singularity does f(z) = 1/(z - 2) have at z = 2?",
            "approach": "Check whether the denominator has a first-power zero at the point.",
            "answer": "A simple pole",
        },
    ],

    "statistics": [
        {
            "description": "Mean/variance/standard-deviation calculation from a small data set.",
            "question": "Find the mean of the data set: 4, 6, 8, 10, 12.",
            "approach": "Add all the values, then divide by the number of values.",
            "answer": "8",
        },
        {
            "description": "Normal distribution, percentile, z-score, or order-statistic probability example.",
            "question": "A test has mean 80 and standard deviation 5. What is the z-score for a score of 90?",
            "options": ["1", "2", "5", "10"],
            "approach": "Use z = (value - mean) / standard deviation.",
            "answer": "B",
        },
        {
            "description": "Hypothesis-test setup involving H0/H1 and significance level.",
            "question": "A company claims its batteries last 12 hours on average. A researcher wants to test whether the batteries last less than 12 hours. What are the null and alternative hypotheses?",
            "approach": "The null usually states equality with the claimed mean; the alternative reflects what the researcher is testing.",
            "answer": "H0: μ = 12; H1: μ < 12",
        },
    ],

    "probability": [
        {
            "description": "Basic probability with favorable outcomes over total outcomes.",
            "question": "A fair six-sided die is rolled once. What is the probability of rolling an even number?",
            "options": ["1/6", "1/3", "1/2", "2/3"],
            "approach": "Count the favorable outcomes and divide by the total number of equally likely outcomes.",
            "answer": "C",
        },
        {
            "description": "Counting/combinatorics example involving arrangements, choices, or repetition.",
            "question": "How many different 3-letter codes can be made using the letters A, B, C, D if repetition is allowed?",
            "approach": "For each of the 3 positions, there are 4 choices. Multiply the choices.",
            "answer": "64",
        },
        {
            "description": "Conditional probability, complement, independence, or inclusion-exclusion example.",
            "question": "A bag has 3 red marbles and 2 blue marbles. If one marble is chosen at random and not replaced, what is the probability that two marbles chosen in a row are both red?",
            "approach": "Multiply the probability of drawing red first by the probability of drawing red second after one red is removed.",
            "answer": "3/10",
        },
    ],

    "sequences_series": [
        {
            "description": "Arithmetic or geometric sequence finite-sum problem.",
            "question": "Find the sum of the first 5 terms of the arithmetic sequence 3, 7, 11, 15, ...",
            "approach": "Find the first 5 terms, then add them together.",
            "answer": "55",
        },
        {
            "description": "Recurrence or pattern-recognition problem using first few terms.",
            "question": "What is the next term in the sequence 2, 6, 18, 54, ...?",
            "options": ["81", "108", "162", "216"],
            "approach": "Identify the pattern from one term to the next.",
            "answer": "C",
        },
        {
            "description": "Floor/log/summation or indexed-series problem where grouping terms helps.",
            "question": "Evaluate the sum: 1 + 2 + 3 + ... + 20.",
            "approach": "Use the arithmetic series formula n(n + 1) / 2.",
            "answer": "210",
        },
    ],

    "general": [
        {
            "description": "Unit conversion or classification problem that does not fit another category.",
            "question": "How many centimeters are in 2.5 meters?",
            "options": ["25", "100", "250", "2500"],
            "approach": "Use the conversion 1 meter = 100 centimeters.",
            "answer": "C",
        },
        {
            "description": "Logic or interpretation problem where careful reading is the main task.",
            "question": "A number is greater than 10 and less than 20. It is divisible by 3 and 5. What is the number?",
            "approach": "List the numbers between 10 and 20 and find the one divisible by both 3 and 5.",
            "answer": "15",
        },
        {
            "description": "Miscellaneous calculation problem with unusual formatting or mixed concepts.",
            "question": "A recipe uses 2 cups of flour for every 3 cups of sugar. If you use 8 cups of flour, how many cups of sugar do you need?",
            "approach": "Set up a proportion using the flour-to-sugar ratio.",
            "answer": "12",
        },
    ],
}

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
    formatted = []
    for label, option in zip(labels, options):
        option = option.strip()
        if re.match(r"^[A-Z]\.\s+", option):
            formatted.append(option)
        else:
            formatted.append(f"{label}. {option}")
    return "\n".join(formatted)


def format_answer(answer) -> str:
    if isinstance(answer, list):
        return ", ".join(str(part) for part in answer)
    return str(answer)


def is_ready_example(item: dict) -> bool:
    return bool(str(item.get("question", "")).strip()) and "answer" in item and str(item.get("answer", "")).strip()


def format_example(item: dict) -> str:
    options_text = format_options(item.get("options"))
    question = item["question"]
    if options_text:
        question = f"{question}\nOptions:\n{options_text}"
    approach = str(item.get("approach", "")).strip()
    approach_text = f"\nApproach:\n{approach}" if approach else ""
    return f"Problem:\n{question}{approach_text}\nAnswer:\n\\boxed{{{format_answer(item['answer'])}}}"


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


def select_curated_examples(topic_name: str, shots: int, is_mcq: Optional[bool] = None) -> list[dict]:
    """Return completed curated examples for a topic.

    Placeholder examples intentionally have empty question/answer fields and are
    skipped until you fill them in.
    """

    if shots <= 0:
        return []
    examples = CURATED_FEW_SHOT_EXAMPLES.get(topic_name, [])
    ready_examples = [example for example in examples if is_ready_example(example)]
    if is_mcq is not None:
        ready_examples = [example for example in ready_examples if bool(example.get("options")) == is_mcq]
    return ready_examples[:shots]


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
