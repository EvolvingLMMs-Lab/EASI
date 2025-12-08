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

## Overview

EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a standardized protocol for the fair evaluation of state-of-the-art proprietary and open-source models.

Key features include:

- Supports the evaluation of **state-of-the-art Spatial Intelligence models**.
- Systematically collects and integrates **evolving Spatial Intelligence benchmarks**.
- Proposes a **standardized testing protocol** to ensure fair evaluation and enable cross-benchmark comparisons.

For the full list of supported models and benchmarks, please refer to  👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**.


## 🗓️ News

🌟 **[2025-12-08]** [EASI v0.1.2](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.2) is released. Major updates include:

- **Expanded model support**  
  Added **5 Spatial Intelligence models** and **1 unified understanding–generation model**:
  - SenseNova-SI 1.1 Series (Qwen2.5-VL-3B / Qwen2.5-VL-7B / Qwen3-VL-8B)
  - SenseNova-SI 1.2 Series (InternVL3-8B)
  - VLM-3R
  - BAGEL-7B-MoT
- **Expanded benchmark support**  
  Added **4 image benchmarks**: STAR-Bench, OmniSpatial, Spatial-Visualization-Benchmark, SPAR-Bench.  

- **LLM-based answer extraction for EASI benchmarks**  
  Added optional LLM-based answer extraction for several EASI benchmarks. You can enable OpenAI judging by:
  ```bash
  --judge gpt-4o-1120
  ```
  which routes to gpt-4o-2024-11-20 for automated evaluation.
---


🌟 **[2025-11-21]**  [EASI v0.1.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.1) is released. Major updates include:

- **Expanded model support**  
  Added **9 Spatial Intelligence models** (total **7 → 16**):
  - SenseNova-SI 1.1 Series (InternVL3-8B / InternVL3-2B)
  - SpaceR-7B
  - VST Series (VST-3B-SFT / VST-7B-SFT)
  - Cambrian-S Series (0.5B / 1.5B / 3B / 7B)


- **Expanded benchmark support**  
  Added **1 image–video benchmark**: VSI-Bench-Debiased.


---


🌟 **[2025-11-07]** [EASI v0.1.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.0) is released. Major updates include:

- **Expanded model support**  
  Supported **7 Spatial Intelligence models**:
  - SenseNova-SI Series (InternVL3-8B / InternVL3-2B)
  - MindCube Series (3B-RawQA-SFT / 3B-Aug-CGMap-FFR-Out-SFT / 3B-Plain-CGMap-FFR-Out-SFT)
  - SpatialLadder-3B
  - SpatialMLLM-4B

- **Expanded benchmark support**  
  Supported **6 Spatial Intelligence benchmarks**:
  - 4 image benchmarks: MindCube, ViewSpatial, EmbSpatial, MMSI (no circular evaluation)
  - 2 image–video benchmarks: VSI-Bench, SITE-Bench

- **Standardized testing protocol**  
  Introduced the EASI testing protocol as described in the [EASI paper](https://arxiv.org/pdf/2508.13142).


## 🛠️ QuickStart
### Installation
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./VLMEvalKit
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


### Evaluation
**General command**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --judge {JUDGE_MODE} --verbose --reuse 
```
See `run.py` for the full list of arguments.

**Example** 

Evaluate `SenseNova-SI-1.2-InternVL3-8B` on `MindCubeBench_tiny_raw_qa`:

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.2-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```
This will use regular expressions to extract the answer. If you want to use an LLM-based judge (e.g., when evaluating SpatialVizBench_CoT),
you can switch the judge to OpenAI:

```
python run.py --data SpatialVizBench_CoT \
              --model {MODEL_NAME} \
              --verbose --reuse --judge gpt-4o-1120
```
Note: to use OpenAI models, you must set the environment variable OPENAI_API_KEY.

## 🖊️ Citation

Spatial intelligence is a rapidly evolving field. Our evaluation scope has expanded beyond GPT-5 to include a broader range of models, leading us to update the paper's title to [*Holistic Evaluation of Multimodal LLMs on Spatial Intelligence*](https://arxiv.org/abs/2508.13142). For consistency, however, the BibTeX below retains the original title for reference.

```bib
@article{easi2025,
  title={Has gpt-5 achieved spatial intelligence? an empirical study},
  author={Cai, Zhongang and Wang, Yubo and Sun, Qingping and Wang, Ruisi and Gu, Chenyang and Yin, Wanqi and Lin, Zhiqian and Yang, Zhitao and Wei, Chen and Shi, Xuanke and Deng, Kewang and Han, Xiaoyang and Chen, Zukai and Li, Jiaqi and Fan, Xiangyu and Deng, Hanming and Lu, Lewei and Li, Bo and Liu, Ziwei and Wang, Quan and Lin, Dahua and Yang, Lei},
  journal={arXiv preprint arXiv:2508.13142},
  year={2025}
}
```