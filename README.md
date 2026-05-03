# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `quick_eval.py` | Fast local one-shot/few-shot prompt testing harness |
| `prompt_strategy.py` | Problem classification, topic hints, and prompt construction |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## Quick Local Prompt Experiments

Use `quick_eval.py` when you want to iterate without running the expensive
notebook cells. It samples a small slice of `data/public.jsonl`, classifies each
problem by math topic, injects topic-specific solving guidance, and can add
one-shot or few-shot examples from the public data.

Preview prompts without loading any model:

```bash
python3 quick_eval.py --limit 3 --shots 1 --preview-only
```

Test the scoring/output plumbing instantly with the mock backend:

```bash
python3 quick_eval.py --limit 5 --shots 2 --backend mock
```

Run a real local Transformers inference pass on a tiny sample:

```bash
python3 quick_eval.py --limit 2 --shots 1 --backend transformers --max-new-tokens 512
```

Classification currently uses transparent keyword rules in `prompt_strategy.py`
for categories such as algebra, geometry, trigonometry, integrals, derivatives,
and gradients. That makes it cheap to edit and inspect before replacing it with
a learned or model-based classifier later.
