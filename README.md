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
- After installation, you can quickly try a SenseNova-SI model with:

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.2-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```

## Overview

EASI is a unified evaluation suite for Spatial Intelligence. It benchmarks state-of-the-art proprietary and open-source multimodal LLMs across a growing set of spatial benchmarks.

Key features include:

- Supports the evaluation of **state-of-the-art Spatial Intelligence models**.
- Systematically collects and integrates **evolving Spatial Intelligence benchmarks**.

As of v0.1.4, EASI supports **21 Spatial Intelligence models** and **17 spatial benchmarks**, and the list is continuously expanding. Full details are available at 👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**.




## 🗓️ News

🌟 **[2025-12-19]** [EASI v0.1.4](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.4) is released. Major updates include:
- **Expanded benchmark support**  
  Added **4 benchmarks**: SPBench, MMSI-Video-Bench, VSI-SUPER-Recall, VSI-SUPER-Count.  

For the full release history and detailed changelog, please see 👉 **[Changelog](docs/CHANGELOG.md)**.

## 🛠️ QuickStart
### Installation
#### Option 1: Local environment
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./VLMEvalKit
```

#### Option 2: Docker-based environment
```
bash dockerfiles/EASI/build_runtime_docker.sh

docker run --gpus all -it --rm \
  -v /path/to/your/data:/mnt/data \
  --name easi-runtime \
  vlmevalkit_EASI:latest \
  /bin/bash
```

### Evaluation
**General command**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --judge {JUDGE_MODE} --verbose --reuse 
```
Please refer to the Configuration section below for the full list of available models and benchmarks
. See run.py for the full list of arguments.

**Example** 

Evaluate `SenseNova-SI-1.2-InternVL3-8B` on `MindCubeBench_tiny_raw_qa`:

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.2-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```
This uses regex-based answer extraction. For LLM-based judging (e.g., on SpatialVizBench_CoT), switch to the OpenAI judge:
```
export OPENAI_API_KEY=YOUR_KEY
python run.py --data SpatialVizBench_CoT \
              --model {MODEL_NAME} \
              --verbose --reuse --judge gpt-4o-1120
```

### Configuration
**VLM Configuration**: During evaluation, all supported VLMs are configured in `vlmeval/config.py`. Make sure you can successfully infer with the VLM before starting the evaluation with the following command `vlmutil check {MODEL_NAME}`. 

**Benchmark Configuration**: The full list of supported Benchmarks can be found in the official VLMEvalKit documentation [VLMEvalKit Supported Benchmarks](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY). 

For the [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard), all EASI benchmarks are summarized in [Supported Models & Benchmarks](docs/Support_bench_models.md). A minimal example of recommended --data settings for EASI is:

| Benchmark   | Evaluation settings          |
|-------------|------------------------------|
| [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | [VSI-Bench_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench.tsv)  |
|             |  [VSI-Bench-Debiased_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench-Debiased.tsv)             |
| [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube)    | [MindCubeBench_tiny_raw_qa](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_tiny_raw_qa.tsv)    |


### Submision

To submit your evaluation results to our [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard):

1. Go to the [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard) page.
2. Click **🚀 Submit here!** to the submission form.
3. Follow the instructions to fill in the submission form, and submit your results.


## 🖊️ Citation

```bib
@article{easi2025,
  title={Holistic Evaluation of Multimodal LLMs on Spatial Intelligence},
  author={Cai, Zhongang and Wang, Yubo and Sun, Qingping and Wang, Ruisi and Gu, Chenyang and Yin, Wanqi and Lin, Zhiqian and Yang, Zhitao and Wei, Chen and Shi, Xuanke and Deng, Kewang and Han, Xiaoyang and Chen, Zukai and Li, Jiaqi and Fan, Xiangyu and Deng, Hanming and Lu, Lewei and Li, Bo and Liu, Ziwei and Wang, Quan and Lin, Dahua and Yang, Lei},
  journal={arXiv preprint arXiv:2508.13142},
  year={2025}
}
```