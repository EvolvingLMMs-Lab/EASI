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

The submission script automates the full pipeline: dataset preparation, evaluation, result collection, and optional leaderboard submission. It supports both **VLMEvalKit** and **lmms-eval** as evaluation backends.

#### Setup

```bash
# Clone the repo and set up the environment
git clone --recurse-submodules https://github.com/EvolvingLMMs-Lab/EASI-upstream.git
cd EASI-upstream
bash scripts/setup.sh
source .venv/bin/activate
```

#### VLMEvalKit Backend (Default)

```bash
# Run EASI-8 core benchmarks on 8 GPUs
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8

# With submission
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

#### lmms-eval Backend

```bash
# Run EASI-8 core benchmarks on 4 GPUs
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model qwen3_vl \
  --model-args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2" \
  --nproc 4

# With submission
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model qwen3_vl \
  --model-args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2" \
  --nproc 4 \
  --submit \
  --submission-configs '{
    "modelName": "Qwen/Qwen3-VL-8B-Instruct",
    "modelType": "instruction",
    "precision": "bfloat16",
    "revision": "main",
    "weightType": "Original",
    "baseModel": "Qwen3",
    "remarks": "Submitted via script"
  }'
```

> **Note:** For lmms-eval, `--model` is the model type (e.g. `qwen3_vl`, `qwen2_5_vl`) and `--model-args` provides the HuggingFace model path and configuration. See the [lmms-eval model list](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) for supported model types.

#### CLI Options

| Option | Description |
|---|---|
| `--model` | **(Required)** For VLMEvalKit: HuggingFace model ID (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`). For lmms-eval: model type (e.g. `qwen3_vl`) |
| `--backend` | Evaluation backend: `vlmevalkit` (default) or `lmms-eval` |
| `--model-args` | **(Required for lmms-eval)** Model arguments (e.g. `pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2`) |
| `--nproc` | Number of GPUs for data parallelism (default: 1). Uses torchrun for VLMEvalKit, accelerate for lmms-eval |
| `--benchmarks` | Comma-separated benchmark keys to run (default: all EASI-8). Use `sitebench` or `site` as shorthand for both `site_image` and `site_video` |
| `--include-extra` | Also run the 4 optional benchmarks (MMSI-Video, OmniSpatial, SPAR-Bench, VSI-Debiased) |
| `--judge` | Override the judge model (VLMEvalKit only; ignored for lmms-eval) |
| `--output-dir` | Output directory (default: `./eval_results`) |
| `--dataset-dir` | Dataset directory (VLMEvalKit only; default: `./datasets`). lmms-eval manages datasets via HuggingFace |
| `--submit` | Submit results to the EASI leaderboard after evaluation. Requires `HF_TOKEN` environment variable or `.env` file |
| `--submission-configs` | JSON string with submission metadata (see fields below) |
| `--rerun` | Force re-evaluation of all benchmarks (skip resume logic) |
| `--no-accelerate` | Use plain Python instead of accelerate for lmms-eval (useful for OOM recovery) |
| `--verbose` | Enable verbose output (VLMEvalKit: `--verbose`; lmms-eval: `--verbosity DEBUG`) |
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
| `remarks` | No | Free-text notes |

#### Output Files

After evaluation, the script generates:

- `eval_results/easi_results.json` — Submission payload with all scores and sub-scores
- `eval_results/easi_results.zip` — Zip archive of all result files for verification
- `eval_results/eval_{model}_{timestamp}.log` — Full evaluation output log

If `--submit` is used and the zip file exceeds 4.5 MB, upload `easi_results.json` and `easi_results.zip` manually at https://easi.lmms-lab.com/submit/ or email them to `easi-lmms-lab@outlook.com`.

#### Resume and Rerun

The script automatically resumes from where it left off. If a run is interrupted or some benchmarks fail, simply rerun the same command — completed benchmarks are skipped.

For lmms-eval, if some benchmarks fail with OOM using accelerate, rerun with `--no-accelerate` to complete the remaining benchmarks in single-GPU mode:

```bash
# Failed benchmarks will be retried without accelerate
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model qwen3_vl \
  --model-args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2" \
  --no-accelerate
```

To force a fresh evaluation (ignore all previous results), add `--rerun`:

```bash
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 --rerun
```

#### More Examples

```bash
# Run specific benchmarks only (VLMEvalKit)
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 4 \
  --benchmarks vsi_bench,blink,sitebench

# Run all benchmarks including extras (lmms-eval)
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model qwen3_vl \
  --model-args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2" \
  --nproc 4 --include-extra

# Force exact_matching judge (VLMEvalKit, no API calls, faster but less accurate for BLINK)
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

### Using lmms-eval Directly

For `lmms-eval`, install the package and run each benchmark individually. Use `accelerate launch` for multi-GPU inference:

```bash
cd lmms-eval
pip install -e .

# Example: run VSI-Bench with 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2 \
    --tasks vsibench_multiimage \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

The EASI-8 benchmark task names in lmms-eval are:

| Benchmark | lmms-eval task name |
|---|---|
| VSI-Bench | `vsibench_multiimage` |
| MMSI-Bench | `mmsi_bench` |
| MindCube-Tiny | `mindcube_tiny` |
| ViewSpatial | `viewspatial` |
| SITE (image) | `site_bench_image` |
| SITE (video) | `site_bench_video_multiimage` |
| BLINK | `blink` |
| 3DSRBench | `3dsrbench` |
| EmbSpatial | `embspatial` |

Optional benchmarks:

| Benchmark | lmms-eval task name |
|---|---|
| MMSI-Video-Bench | `mmsi_video_u50` |
| OmniSpatial (Manual CoT) | `omnispatial_test` |
| SPAR-Bench | `sparbench` |
| VSI-Debiased | `vsibench_debiased_multiimage` |

> **Note:** Always pass `--log_samples` when running lmms-eval directly. The per-sample JSONL files are needed for SiteBench combined scoring and for result submission.

For the full list of supported model types, see the [lmms-eval models directory](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models).

## Submission

Once evaluation is complete, you can submit results in one of three ways:

1. **Automated** — Use `--submit` flag with the submission script (see above)
2. **Web form** — Upload `easi_results.json` and `easi_results.zip` at https://easi.lmms-lab.com/submit/
3. **Email** — Send result files to `easi-lmms-lab@outlook.com`

For questions, please contact `easi-lmms-lab@outlook.com`.
