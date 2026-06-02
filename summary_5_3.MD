# Progress Report: LLM Inference Optimization & Evaluation Pipeline

## 1. Initial Baseline & Bottleneck Identification

We began by running the raw model on the full public dataset using standard generation parameters:

- `max_tokens = 8192`
- `temperature = 0.6`

This baseline achieved an overall score of **~53.3%** on the private set, but required roughly **3 hours** to complete.

### Key Bottlenecks Identified

- **Intrinsic Capacity Limitations (Genuine Math Errors)**
  - As expected from a 4B parameter model, a baseline percentage of failures are genuine mathematical errors. These include choosing the wrong theorem, misunderstanding complex problem setups, or lacking the latent knowledge required for advanced graduate-level topics.

- **Format Anxiety / Verification Loops**
  - The model often solved problems correctly early in generation, but then spent thousands of tokens second-guessing formatting and repeatedly verifying the final answer.

- **Arithmetic Spirals (Over-Computation)**
  - Instead of leaving answers in exact symbolic form (e.g., fractions or expressions involving $\pi$), the model attempted lengthy decimal expansions and manual long-division computations.
  - This consumed hundreds of unnecessary tokens and frequently introduced arithmetic errors.

- **Constraint Degradation ("The Missing Box")**
  - On longer reasoning traces, attention quality degraded near the end of generation.
  - The model would occasionally forget to wrap the final answer in the required `\boxed{}` syntax, causing automatic judger failure despite correct mathematics.

- **Repetition Traps**
  - The model occasionally entered autoregressive loops, repeating identical algebraic manipulations or restating variables multiple times.
  - These loops inflated token usage without improving reasoning quality.

### Immediate Action

To enable faster iteration and debugging:

- Isolated a **200-question validation subset**
- Reduced `max_tokens` from `8192 → 4096`

This acted as a practical “circuit breaker” against runaway generations and infinite reasoning loops. Our reasoning is that ideally, any correct response should be under 4096 tokens. If it is taking longer than that, the model is most likely rambling nonsense.

---

## 2. Establishing a Deterministic Baseline (Temperature Ablation)

Before modifying prompts, we evaluated multiple temperatures on the 200-question subset using the new `4096` token cap. Aside from the baseline T = 0.6, we test T = 0.3, 0.1, and 0.

### Result

- **Best single-pass performance:** `T = 0.0` (greedy decoding)

While greedy decoding is standard for math reasoning (for single passes), explicitly validating this experimentally was important for rigorous benchmarking.

### Rationale

Using `T = 0.0` eliminated stochasticity from random sampling (RNG), ensuring that any future performance gains could be attributed directly to prompt engineering rather than sampling variance.

---

## 3. Prompt Engineering & Few-Shot Injection

Using the deterministic `T = 0.0` baseline, we iteratively refined the system prompt to address observed behavioral failures.

### Prompt Modifications

- **Conciseness Rule**
  - Including rules about keeping thinking concise and computation focused
  - Explicitly prohibited formatting debates inside reasoning traces.
  - Introduced a “Decisiveness Protocol” to reduce recursive self-verification.

- **Targeted Few-Shot Examples**
  - Injected curated multi-shot demonstrations.

### Result

We observed:

- Significant improvement on the 200-question validation set
- Particularly strong gains in the MCQ category
- Shorter and cleaner reasoning traces
- Reduced token waste

---

## 4. Full-Set Evaluation & the Token-Truncation Trade-off

We then evaluated the optimized configuration:

- `T = 0.0`
- Few-shot prompting enabled
- `max_tokens = 4096`

across both the full public and private datasets, requiring roughly **1.5 hours** for each.

### Results

| Dataset | Previous | Optimized |
|---|---|---|
| Public | 53.1% | 55.8% |
| Private | 53.3% | 52.6% |

### Diagnosis: Context Window Limitation

The slight private-set regression appears attributable to the reduced token budget.

While the `4096` token cap successfully:

- eliminated most formatting rambling,
- dramatically improved inference speed,
- and stabilized outputs,

it also introduced a hard truncation ceiling.

We suspect for the most difficult problems (roughly the top ~5%), the model occasionally required more than 4096 tokens to complete a full derivation (including any of its rambling and conversational text, as this was not completely eliminated with the new prompts), causing valid reasoning chains to terminate prematurely.

(can maybe rerun the original baseline on 4096 max tokens, and then observe a true gain and not have to talk about any of this)

### Trade-off

We effectively exchanged:

- a small reduction in accuracy on extremely long reasoning tasks

for:

- substantially faster inference,
- improved formatting stability,
- and dramatically better iteration speed.

---

## 5. Dataset & Judger Discoveries ("False Negatives")

Failure analysis revealed that the model is performing better than the raw metrics suggest.

Many incorrect evaluations are actually **false negatives** caused by weaknesses in the given evaluation pipeline itself.

### Issues Identified

- **Strict String-Matching Failures**
  - Mathematically equivalent expressions are marked incorrect due to formatting differences.

- **Malformed Gold Answers**
  - Some dataset labels appear incorrect or inconsistent.

## 6. Next Steps: Advanced Inference Scaling (Majority Voting)

With prompt-level guardrails now reducing formatting failures and sequence-related panic behaviors, the next step is to implement:

- **Majority Voting / Self-Consistency**
- Configuration: `T = 0.6`, `n = 5`

Reintroducing moderate temperature (`T = 0.6`) allows the model to explore multiple reasoning trajectories per problem, from which the majority answer will be taken

### Expected Benefit

This approach should improve robustness against:

- formatting variance,
- token-limit failures,
- and brittle answer canonicalization.