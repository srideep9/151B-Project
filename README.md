# CSE 151B Math Reasoning Competition Submission

This branch exposes a single end-to-end inference entry point:

```python
from mainInference import run_inference

run_inference()
```

Calling `run_inference()` loads the model, runs inference on `data/private.jsonl`,
applies our post-processing, and writes the final submission CSV.

## Hardware And Runtime

This pipeline was designed for a CUDA GPU environment with vLLM.

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Precision/loading: BF16, no BitsAndBytes quantization
- GPU used for selected submission: `1 x NVIDIA A100-SXM4-80GB`
- Approximate selected submission runtime: `1h 53m 1s`
- Selected submission max generation tokens: `8192`
- Selected submission max model length: `max_tokens + 4096`
- Selected submission max parallel sequences: `128`
- Selected submission majority voting: `--num-outputs 5`
- Selected submission temperature/top-p: `0.6` / `0.95`

The first run may take longer because Hugging Face/vLLM need to download and
cache model weights.

## Setup

Install dependencies in a CUDA Linux environment:

```bash
pip install vllm transformers pandas tqdm numpy sympy antlr4-python3-runtime==4.11.1
```

The model weights are downloaded automatically from Hugging Face:

```text
Qwen/Qwen3-4B-Thinking-2507
```

If your environment requires Hugging Face authentication, run:

```bash
huggingface-cli login
```

Keep the dataset files in the repository under `data/`:

```text
data/private.jsonl
data/public.jsonl
```

## Reproduce Submission

To reproduce our selected submission, run the private set with 5-way majority
voting:

Python API:

```python
from mainInference import run_inference

csv_path = run_inference(
    output_path="results/inference.jsonl",
    num_outputs=5,
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192,
    max_num_seqs=128,
)
print(csv_path)
```

Command line:

```bash
python mainInference.py \
  --data-path data/private.jsonl \
  --output-path results/inference.jsonl \
  --num-outputs 5 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 8192 \
  --max-num-seqs 128
```

Outputs:

```text
results/inference.jsonl
results/inference.csv
```

The CSV has the required submission format:

```text
id,response
```

## Methodology

We use Qwen3-4B-Thinking through vLLM with BF16 weights and no quantization. The
system prompts ask the model to reason inside `<think>` tags and require a final
answer inside a single `\boxed{}` expression.
Multiple-choice and free-response questions use separate prompt histories and
few-shot examples.

The selected submission uses 5-way mathematical majority voting. The model
generates five traces for each problem at temperature `0.6`. Final answers are
extracted with the provided `Judger`, grouped by symbolic/numeric equivalence,
and the shortest trace from the largest equivalent answer group is selected.
This keeps the post-processing aligned with the competition judging logic.

We also tried several more complicated training and inference approaches,
including distillation/fine-tuning-style runs and more elaborate inference
variants. In our Kaggle submissions, this simpler BF16 baseline with 5-way
majority voting was the best-performing selected submission.
