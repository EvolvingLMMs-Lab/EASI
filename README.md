<div align="center">
  <img src="assets/banner.jpg"/>

  <b>EASI: Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</b>

  English | [简体中文](README_CN.md) 
</div>


<p align="center">
    <a href="https://arxiv.org/abs/2508.13142" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-EASI-red?logo=arxiv" height="20" />
    </a>
    <!-- <a href="https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard" target="_blank">
        <img alt="Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_EASI-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a> -->
    <a href="https://easi.lmms-lab.com/leaderboard/" target="_blank">
        <img alt="Leaderboard" src="https://img.shields.io/badge/🏆_EASI-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/EvolvingLMMs-Lab/EASI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/EvolvingLMMs-Lab/EASI?style=flat"></a>
</p>

## TL;DR

- EASI is a unified evaluation suite for Spatial Intelligence in multimodal LLMs.
- EASI supports **two evaluation backends**: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
- After installation, you can run all EASI-8 benchmarks with a single command:

**Using VLMEvalKit backend (default):**
```bash
python scripts/submissions/run_easi_eval.py \
  --model sensenova/SenseNova-SI-1.3-InternVL3-8B \
  --nproc 4
```

**Using lmms-eval backend:**
```bash
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model internvl2 \
  --model-args "pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B" \
  --nproc 4
```

> Under the hood, EASI wraps [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with a unified CLI. See the respective repos for advanced usage and adding custom models.

## Overview

EASI is a unified evaluation suite for Spatial Intelligence. It benchmarks state-of-the-art proprietary and open-source multimodal LLMs across a growing set of spatial benchmarks.

- **Comprehensive Support**: Currently EASI([v0.2.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.1)) supports **23 Spatial Intelligence models** and **27 spatial benchmarks**.
- **Dual Backends**:
  - **VLMEvalKit**: Rich model zoo with built-in judging capabilities.
  - **lmms-eval**: Lightweight, accelerate-based distributed evaluation.

Full details are available at 👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**. EASI also provides transparent 👉 **[Benchmark Verification](docs/Benchmark_Verification.md)** against official scores.


## 🗓️ News

🌟 **[2026-02-09]** [EASI v0.2.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.1) is released. Major updates include:
- **Expanded benchmark support**: Added ERIQ and OSI-Bench.
- **Bug fixes**: Fixed VLMEvalKit evaluation issues on MuirBench.
- **Benchmark verification**: Added more lmms-eval benchmark verification entries.

🌟 **[2026-01-16]** [EASI v0.2.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.0) is released. Major updates include:
- **New Backend Support**: Integrated lmms-eval alongside VLMEvalKit, offering flexible evaluation options.
- **Expanded benchmark support**: Added DSR-Bench. 


For the full release history and detailed changelog, please see 👉 **[Changelog](docs/CHANGELOG.md)**.

## 🛠️ QuickStart
### Installation

#### Option 1: Local environment (Recommended)

The setup script installs both evaluation backends (VLMEvalKit and lmms-eval) with pinned dependencies:

```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
bash scripts/setup.sh
source .venv/bin/activate
```

This creates a Python 3.11 virtual environment with both backends, flash-attn, and all required dependencies. See [scripts/setup.sh](scripts/setup.sh) for details.

#### Option 2: Docker-based environment
```
bash dockerfiles/EASI/build_runtime_docker.sh

docker run --gpus all -it --rm \
  -v /path/to/your/data:/mnt/data \
  --name easi-runtime \
  VLMEvalKit_EASI:latest \
  /bin/bash
```

### Evaluation

EASI provides a unified evaluation script that supports both VLMEvalKit and lmms-eval backends. The script handles dataset preparation, evaluation, result collection, and optional leaderboard submission.

---

#### Using the Unified Evaluation Script (Recommended)

**VLMEvalKit backend (default):**
```bash
# Run EASI-8 core benchmarks on 4 GPUs
python scripts/submissions/run_easi_eval.py \
  --model sensenova/SenseNova-SI-1.3-InternVL3-8B \
  --nproc 4
```

**lmms-eval backend:**
```bash
# Run EASI-8 core benchmarks on 4 GPUs
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model internvl2 \
  --model-args "pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B" \
  --nproc 4
```

**With automated submission:**
```bash
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model internvl2 \
  --model-args "pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B" \
  --nproc 4 \
  --submit \
  --submission-configs '{
    "modelName": "sensenova/SenseNova-SI-1.3-InternVL3-8B",
    "modelType": "instruction",
    "precision": "bfloat16"
  }'
```

**More options:**
```bash
# Run specific benchmarks only
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --benchmarks vsi_bench,blink,sitebench

# Include extra benchmarks (MMSI-Video, OmniSpatial, SPAR-Bench, VSI-Debiased)
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 --include-extra

# Force re-evaluation (ignore previous results)
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 --rerun

# lmms-eval OOM recovery: complete failed benchmarks in single-GPU mode
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model qwen3_vl \
  --model-args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2" \
  --no-accelerate
```

Full CLI options and submission config details at 👉 **[Submission Guide](docs/Submit_results.md)**.

---

#### Using Backends Directly

For advanced usage or custom model integration, you can also call the backends directly:

**VLMEvalKit:**
```bash
cd VLMEvalKit/
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.3-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```

**lmms-eval:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 -m lmms_eval \
    --model internvl2 \
    --model_args=pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B \
    --tasks vsibench_multiimage \
    --batch_size 1 --log_samples --output_path ./logs/
```

For more details, refer to the [VLMEvalKit documentation](VLMEvalKit/README.md) and [lmms-eval documentation](lmms-eval/README.md).

---

### Configuration

- **Supported Models & Benchmarks**: Summarized in [Supported Models & Benchmarks](docs/Support_bench_models.md).
- **VLMEvalKit Models**: Defined in `vlmeval/config.py`. Verify inference with `vlmutil check {MODEL_NAME}`.
- **lmms-eval Models**: Supports various model types (`qwen2_5_vl`, `llava`, `internvl2`, etc.). See the [lmms-eval models directory](lmms-eval/lmms_eval/models/).

### Submission

You can submit your evaluation results at 👉 **[EASI Leaderboard Submission](https://easi.lmms-lab.com/submit/)**.

Full details and file format examples are available at 👉 **[Submission Guide](docs/Submit_results.md)**.

## 🤝 Contribution

EASI is an open and evolving evaluation suite. We warmly welcome community contributions, including:
- New spatial benchmarks
- New model baselines
- Evaluation tools

If you are interested in contributing, or have questions about integration, please contact us at
📧 easi-lmms-lab@outlook.com


## 🖊️ Citation

```bib
@article{easi2025,
  title={Holistic Evaluation of Multimodal LLMs on Spatial Intelligence},
  author={Cai, Zhongang and Wang, Yubo and Sun, Qingping and Wang, Ruisi and Gu, Chenyang and Yin, Wanqi and Lin, Zhiqian and Yang, Zhitao and Wei, Chen and Shi, Xuanke and Deng, Kewang and Han, Xiaoyang and Chen, Zukai and Li, Jiaqi and Fan, Xiangyu and Deng, Hanming and Lu, Lewei and Li, Bo and Liu, Ziwei and Wang, Quan and Lin, Dahua and Yang, Lei},
  journal={arXiv preprint arXiv:2508.13142},
  year={2025}
}
```
