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
- Default GPU setting: one CUDA-visible GPU
- Default max generation tokens: `8192`
- Default max model length: `max_tokens + 4096`
- Default max parallel sequences: `64`
- Majority voting: supported through `--num-outputs`; default is `1`

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

Python API:

```python
from mainInference import run_inference

csv_path = run_inference()
print(csv_path)
```

Command line:

```bash
python mainInference.py \
  --data-path data/private.jsonl \
  --output-path results/inference.jsonl \
  --num-outputs 1 \
  --temperature 0 \
  --top-p 0.95 \
  --max-tokens 8192 \
  --max-num-seqs 64
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

The pipeline includes mathematical majority voting. When `--num-outputs` is
greater than 1, the model generates multiple traces for each problem. Final
answers are extracted with the provided `Judger`, grouped by symbolic/numeric
equivalence, and the shortest trace from the largest equivalent answer group is
selected. This keeps the post-processing aligned with the competition judging
logic.

## Optional Majority-Vote Run

To run with voting, increase `--num-outputs` and use a nonzero temperature:

```bash
python mainInference.py \
  --data-path data/private.jsonl \
  --output-path results/inference_vote.jsonl \
  --num-outputs 5 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 8192 \
  --max-num-seqs 64
```
