# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `quick_eval.py` | Fast local one-shot/few-shot prompt testing harness |
| `prepare_numina_teacher_sft.py` | Builds a 5k NuminaMath-CoT teacher-SFT JSONL using the inference prompt path |
| `compare_results.py` | Compares two `quick_eval.py` result files |
| `prompt_strategy.py` | Problem classification, topic hints, and prompt construction |
| `routing_audit.py` | Creates manual review files for checking topic routing quality |
| `score_routing_audit.py` | Scores a manually labeled routing audit file |
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
python3 quick_eval.py --limit 5 --shots 2 --prompt-mode routed --backend mock
```

Create a routing audit file for manual topic-label review:

```bash
python3 routing_audit.py --limit 50 --output results/routing_audit.jsonl
```

After filling in `correct_topic` for those rows, score the router:

```bash
python3 score_routing_audit.py --audit results/routing_audit.jsonl
```

Run a real local Transformers inference pass on a tiny sample:

```bash
python3 quick_eval.py --limit 2 --shots 1 --prompt-mode routed --backend transformers --max-new-tokens 512
```

Classification currently uses transparent keyword rules in `prompt_strategy.py`
for categories such as algebra, geometry, trigonometry, integrals, derivatives,
gradients, differential equations, complex analysis, statistics, probability,
and sequences/series. That makes it cheap to edit and inspect before replacing
it with a learned or model-based classifier later.

## Checking Routed Prompting

Use this workflow to test whether topic routing improves model accuracy. 
Run both commands on the same dataset slice:

```bash
python3 quick_eval.py --limit 25 --prompt-mode baseline --backend transformers --output results/baseline_25.jsonl
python3 quick_eval.py --limit 25 --shots 1 --prompt-mode routed --backend transformers --quantization 4bit --output results/routed_25.jsonl
```

Then compare the result files:

```bash
python3 compare_results.py --base results/baseline_25.jsonl --candidate results/routed_25.jsonl
```

`baseline` uses the starter-style prompt. `routed` adds the topic-specific
prompt hint and same-topic few-shot examples. Increase `--limit` only after the
small run looks healthy.

## NuminaMath Teacher SFT Data

Prepare 5,000 NuminaMath-CoT rows with the same system/user prompt format used
at inference time:

```bash
python3 prepare_numina_teacher_sft.py
```

The script downloads/caches NuminaMath-CoT parquet shards from Hugging Face,
keeps rows whose assistant solution already has a boxed answer or can be safely
converted to one, and writes chat JSONL to `data/numina_teacher_sft_5k.jsonl`.

Preview real rows before writing the file:

```bash
python3 prepare_numina_teacher_sft.py --limit 2 --max-scan 50 --preview 1
```
