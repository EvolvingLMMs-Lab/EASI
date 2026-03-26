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

For `lmms-eval`, it is similar. The benchmark names can be found in:

- https://github.com/EvolvingLMMs-Lab/EASI/blob/main/docs/Support_bench_models.md

Once we receive these materials, we can proceed with leaderboard integration.

For questions, please contact `easi-lmms-lab@outlook.com`.
