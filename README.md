# EASI

<b>Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</b>

English | [简体中文](README_CN.md) 

<p align="center">
    <a href="https://arxiv.org/abs/2508.13142" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-EASI-red?logo=arxiv" height="20" />
    </a>
    <a href="https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard" target="_blank">
        <img alt="Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_EASI-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/EvolvingLMMs-Lab/EASI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/EvolvingLMMs-Lab/EASI?style=flat"></a>
</p>

## TL;DR

- EASI is a unified evaluation suite for Spatial Intelligence in multimodal LLMs.
- EASI supports **two evaluation backends**: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
- After installation, you can quickly try a SenseNova-SI model with:

**Using EASI (backend=VLMEvalKit):**
```bash
cd VLMEvalKit/
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.3-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```

**Using EASI (backend=lmms-eval):**
```bash
lmms-eval --model qwen2_5_vl \
          --model_args pretrained=sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B \
          --tasks site_bench_image \
          --batch_size 1 \
          --log_samples \
          --output_path ./logs/
```

## Overview

EASI is a unified evaluation suite for Spatial Intelligence. It benchmarks state-of-the-art proprietary and open-source multimodal LLMs across a growing set of spatial benchmarks.

- **Comprehensive Support**: Currently EASI([v0.2.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.0)) supports **23 Spatial Intelligence models** and **25 spatial benchmarks**.
- Dual Backends:
  - **VLMEvalKit**: Rich model zoo with built-in judging capabilities.
  - **lmms-eval**: Lightweight, accelerate-based distributed evaluation.

Full details are available at 👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**. EASI also provides transparent 👉 **[Benchmark Verification](docs/Benchmark_Verification.md)** against official scores.


## 🗓️ News

🌟 **[2026-01-16]** [EASI v0.2.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.0) is released. Major updates include:
- **New Backend Support**: Integrated lmms-eval alongside VLMEvalKit, offering flexible evaluation options.
- **Expanded benchmark support**: Added DSR-Bench. 


For the full release history and detailed changelog, please see 👉 **[Changelog](docs/CHANGELOG.md)**.

## 🛠️ QuickStart
### Installation

EASI provides two evaluation backends. You can install one or both depending on your needs.

#### Option 1: Local environment (backend=VLMEvalKit)
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./VLMEvalKit
```

#### Option 2: Local environment (backend=lmms-eval)
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./lmms-eval spacy
# Recommended Dependencies
# Use "torch==2.7.1", "torchvision==0.22.1" in pyproject.toml (this works with most models)
# Install flash-attn for faster inference
pip install flash-attn --no-build-isolation
```

#### Option 3: Docker-based environment
```
bash dockerfiles/EASI/build_runtime_docker.sh

docker run --gpus all -it --rm \
  -v /path/to/your/data:/mnt/data \
  --name easi-runtime \
  VLMEvalKit_EASI:latest \
  /bin/bash
```

### Evaluation

EASI supports two evaluation backends. Choose the one that best fits your needs.

---

#### Backend 1: VLMEvalKit

**General command**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --judge {JUDGE_MODE} --verbose --reuse 
```
Please refer to the Configuration section below for the full list of available models and benchmarks. See `run.py` for the full list of arguments.

**Example** 

Evaluate `SenseNova-SI-1.3-InternVL3-8B` on `MindCubeBench_tiny_raw_qa`:

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.3-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```
This uses regex-based answer extraction. For LLM-based judging (e.g., on SpatialVizBench_CoT), switch to the OpenAI judge:
```bash
export OPENAI_API_KEY=YOUR_KEY
python run.py --data SpatialVizBench_CoT \
              --model {MODEL_NAME} \
              --verbose --reuse --judge gpt-4o-1120
```

---

#### Backend 2: lmms-eval

lmms-eval provides accelerate-based distributed evaluation with support for multi-GPU inference.

**General command**
```bash
lmms-eval --model {MODEL_TYPE} \
          --model_args pretrained={MODEL_PATH} \
          --tasks {TASK_NAME} \
          --batch_size 1 \
          --log_samples \
          --output_path ./logs/
```

**Example: Single GPU**

Evaluate `SenseNova-SI-1.1-Qwen2.5-VL-3B` on `site_bench_image`:

```bash
lmms-eval --model qwen2_5_vl \
          --model_args pretrained=sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B \
          --tasks site_bench_image \
          --batch_size 1 \
          --log_samples \
          --output_path ./logs/
```

**Example: Multi-GPU with accelerate**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B,attn_implementation=flash_attention_2 \
    --tasks site_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

**List available tasks**
```bash
lmms-eval --tasks list
```

For more details on lmms-eval usage, refer to the documentation in [lmms-eval/docs/](lmms-eval/docs/), including [model guide](lmms-eval/docs/model_guide.md), [task guide](lmms-eval/docs/task_guide.md), and [run examples](lmms-eval/docs/run_examples.md).

---

### Configuration

**EASI (backend=VLMEvalKit)**
- **Models**: Defined in `vlmeval/config.py`. Verify inference with `vlmutil check {MODEL_NAME}`.
- **Benchmarks**: Full list of supported Benchmarks at [VLMEvalKit Supported Benchmarks](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY). 
- **EASI Specifics**: For [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard), related benchmarks are summarized in [Supported Models & Benchmarks](docs/Support_bench_models.md). 

**EASI (backend=lmms-eval)**
- **Models**: lmms-eval supports various model types including `qwen2_5_vl`, `llava`, `internvl2`, and more. Use `--model_args` to specify model parameters like `pretrained`, `attn_implementation`, etc.

- **Tasks**: Tasks are defined in `lmms-eval/lmms_eval/tasks/`. To list all available tasks:
  ```bash
  lmms-eval --tasks list
  ```

  Example tasks for spatial intelligence evaluation:
  | Task Name | Description |
  |-----------|-------------|
  | `site_bench_image` | SITE-Bench image evaluation |
  | `site_bench_video` | SITE-Bench video evaluation |

  For more details on lmms-eval usage, refer to the [lmms-eval documentation](lmms-eval/README.md).


### Submision

To submit your evaluation results to our [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard):

1. Go to the [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard) page.
2. Click **🚀 Submit here!** to the submission form.
3. Follow the instructions to fill in the submission form, and submit your results.

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
