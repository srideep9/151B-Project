# CSE 151B Math Reasoning Competition Submission

This branch exposes a single end-to-end inference entry point:

```python
from mainInference import run_inference

run_inference()
```

Calling `run_inference()` loads the model, runs inference on `data/private.jsonl`,
applies majority-vote post-processing, and writes the final submission CSV.

## Hardware And Runtime

This pipeline was designed for a CUDA GPU environment and tested with vLLM.

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Loading mode: vLLM with BitsAndBytes quantization
- Default GPU setting: one CUDA-visible GPU
- Default max generation tokens: `4096`
- Default max model length: `8192`
- Default generations per problem: `1`

When `--num-outputs` is greater than 1, the script generates multiple candidate
traces per problem and uses mathematical majority voting to choose the final
response.

## Setup

Install the required packages in a CUDA Linux environment:

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
  --max-tokens 4096 \
  --max-model-len 8192 \
  --max-num-seqs 32
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

We use Qwen3-4B-Thinking through vLLM with separate prompts for free-response and
multiple-choice questions. The prompt asks the model to reason inside `<think>`
tags and put the final answer inside a single `\boxed{}` expression.

The branch also includes majority-voting support. For each problem, the model can
produce multiple traces. The final answers are extracted with the provided
`Judger`, grouped by mathematical equivalence, and the shortest trace from the
largest equivalent answer group is selected. This keeps answer selection aligned
with the same symbolic/numeric comparison logic used by the competition judging
code.

## Notes

The default command above reproduces the branch defaults. To enable majority
voting, increase `--num-outputs`, for example:

```bash
python mainInference.py \
  --data-path data/private.jsonl \
  --output-path results/inference_vote.jsonl \
  --num-outputs 5 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --max-model-len 8192 \
  --max-num-seqs 32
```
