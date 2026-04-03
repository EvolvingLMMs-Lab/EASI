# EASI Leaderboard Submission

Thank you for your interest in contributing results to the EASI leaderboard.

To include your method in the leaderboard, please first evaluate your model on the EASI-8 benchmarks, then send us the result files and the evaluation setup.

## What to Submit

Please provide:

1. Per-question results from your model, including the extracted answers.
2. Aggregated performance files, including accuracy or related metrics.
3. The exact evaluation setup used for reproduction.

For different evaluation backends, the expected files are typically in the following form. These are only examples, and the exact file names do not need to match:

- `VLMEvalKit`:
  - per-question results: `(model_name)_VSI-Bench_32frame_extract_matching.xlsx`
  - aggregated results: `(model_name)_VSI-Bench_32frame_acc.csv`
- `lmms-eval`:
  - aggregated results: `(model_name)_20260209_222541_results.json`
  - per-question results: `(model_name)_20260209_222541_samples_embspatial.jsonl`

If you did not use EASI for evaluation, please still provide equivalent files with the same information as above.

## Evaluation Setup

For open-source models, please also provide the key settings needed for reproduction, including:

- model name / checkpoint
- backend used (`VLMEvalKit` or `lmms-eval`)
- important evaluation settings

For benchmark names and backend settings, please refer to:

- https://github.com/EvolvingLMMs-Lab/EASI/blob/main/docs/Support_bench_models.md

## Required Benchmarks

At minimum, please include results on the following EASI-8 benchmarks:

- `VSI-Bench`
- `MMSI-Bench`
- `MindCube-Tiny`
- `ViewSpatial`
- `SITE`
- `BLINK`
- `3DSRBench`
- `EmbSpatial`

## Optional Benchmarks

The following benchmarks are optional but encouraged:

- `MMSI-Video-Bench`
- `OmniSpatial (Manual CoT)`
- `SPAR-Bench`
- `VSI-Debiased`

## How to Run EASI-8

### Using the Submission Script (Recommended)

The submission script automates the full pipeline: dataset preparation, evaluation, result collection, and optional leaderboard submission. Currently, this script only supports VLMEvalKit as evaluation backend.

#### Setup

```bash
# Clone the repo and set up the environment
git clone --recurse-submodules https://github.com/EvolvingLMMs-Lab/EASI-upstream.git
cd EASI-upstream
bash scripts/setup.sh
source .venv/bin/activate
```

#### Basic Usage

```bash
# Run EASI-8 core benchmarks on 8 GPUs
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8
```

#### Full Usage with Submission

```bash
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 \
  --submit \
  --submission-configs '{
    "modelName": "Qwen/Qwen2.5-VL-7B-Instruct",
    "modelType": "instruction",
    "precision": "bfloat16",
    "revision": "main",
    "weightType": "Original",
    "baseModel": "",
    "remarks": "Submitted via script"
  }'
```

#### CLI Options

| Option | Description |
|---|---|
| `--model` | **(Required)** Model name — HuggingFace ID (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) or VLMEvalKit model name |
| `--nproc` | Number of GPUs for data parallelism via torchrun (default: 1) |
| `--benchmarks` | Comma-separated benchmark keys to run (default: all EASI-8). Use `sitebench` or `site` as shorthand for both `site_image` and `site_video` |
| `--include-extra` | Also run the 4 optional benchmarks (MMSI-Video, OmniSpatial, SPAR-Bench, VSI-Debiased) |
| `--judge` | Override the judge model for all benchmarks (e.g. `exact_matching` or `gpt-4o-1120`). If not specified, uses VLMEvalKit per-benchmark defaults. BLINK is additionally re-evaluated with `gpt-4o-1120` by default |
| `--output-dir` | Output directory (default: `./eval_results`) |
| `--dataset-dir` | Dataset directory (default: `./datasets`) |
| `--submit` | Submit results to the EASI leaderboard after evaluation. Requires `HF_TOKEN` environment variable or `.env` file |
| `--submission-configs` | JSON string with submission metadata (see fields below) |
| `--verbose` | Pass `--verbose` to VLMEvalKit (prints per-sample model responses) |
| `--no-rich` | Disable the rich terminal UI (for CI/non-interactive terminals) |

#### Submission Config Fields

When using `--submit`, the following fields can be set via `--submission-configs`:

| Field | Required | Description |
|---|---|---|
| `modelName` | Yes | HuggingFace model ID in `org/model` format |
| `modelType` | Yes | `pretrained`, `finetuned`, `instruction`, or `rl` |
| `precision` | Yes | `bfloat16`, `float16`, `float32`, or `int8` |
| `revision` | No | Model revision (default: `main`) |
| `weightType` | No | `Original`, `Delta`, or `Adapter` |
| `baseModel` | No | Required for Delta/Adapter weights |
| `backend` | No | Defaults to `vlmevalkit` |
| `remarks` | No | Free-text notes |

#### Output Files

After evaluation, the script generates:

- `eval_results/easi_results.json` — Submission payload with all scores and sub-scores
- `eval_results/easi_results.zip` — Zip archive of all result files for verification
- `eval_results/eval_{model}_{timestamp}.log` — Full VLMEvalKit output log

If `--submit` is used and the zip file exceeds 4.5 MB, upload `easi_results.json` and `easi_results.zip` manually at https://easi.lmms-lab.com/submit/ or email them to `easi-lmms-lab@outlook.com`.

#### Examples

```bash
# Run specific benchmarks only
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 4 \
  --benchmarks vsi_bench,blink,sitebench

# Run all benchmarks including extras
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 \
  --include-extra

# Force exact_matching judge (no API calls, faster but less accurate for BLINK)
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 \
  --judge exact_matching
```

### Using VLMEvalKit Directly

For `VLMEvalKit`, please first add your model inference code under `VLMEvalKit/vlmeval/vlm/`, then follow the top-level `README.md` to prepare the environment, and run:

```bash
cd VLMEvalKit

python run.py --data \
              MindCubeBench_tiny_raw_qa \
              ViewSpatialBench \
              EmbSpatialBench \
              MMSIBench_wo_circular \
              VSI-Bench_32frame \
              SiteBenchImage \
              SiteBenchVideo_32frame \
              BLINK \
              3DSRBench \
              --model {your_model} \
              --verbose --reuse --judge gpt-4o-1120
```

### Using lmms-eval

For `lmms-eval`, it is similar. The benchmark names can be found in:

- https://github.com/EvolvingLMMs-Lab/EASI/blob/main/docs/Support_bench_models.md

## Submission

Once evaluation is complete, you can submit results in one of three ways:

1. **Automated** — Use `--submit` flag with the submission script (see above)
2. **Web form** — Upload `easi_results.json` and `easi_results.zip` at https://easi.lmms-lab.com/submit/
3. **Email** — Send result files to `easi-lmms-lab@outlook.com`

For questions, please contact `easi-lmms-lab@outlook.com`.
