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

## 🗓️ News

🌟 **[2025-11-21]**
[EASI v0.1.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.1) is released. Major updates include:

- **Expanded model support**  
  Added **9 new Spatial Intelligence models**, increasing total from **7 → 16**:
    - **SenseNova-SI 1.1 Series**  
        - [SenseNova-SI-1.1-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-8B)  
        - [SenseNova-SI-1.1-InternVL3-2B](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-2B)
    - SpaceR: [SpaceR-7B](https://huggingface.co/RUBBISHLIKE/SpaceR)
    - VST Series: [VST-3B-SFT](https://huggingface.co/rayruiyang/VST-3B-SFT), [VST-7B-SFT](https://huggingface.co/rayruiyang/VST-7B-SFT)
    - Cambrian-S series: [Cambrian-S-0.5B](https://huggingface.co/nyu-visionx/Cambrian-S-0.5B), [Cambrian-S-1.5B](https://huggingface.co/nyu-visionx/Cambrian-S-1.5B), [Cambrian-S-3B](https://huggingface.co/nyu-visionx/Cambrian-S-3B), [Cambrian-S-7B](https://huggingface.co/nyu-visionx/Cambrian-S-7B)

- **Expanded benchmark support**  
  Added **1 new image–video benchmark**, increasing total from **6 → 7**:
    - [**VSI-Bench-Debiased**](https://vision-x-nyu.github.io/thinking-in-space.github.io/)

---


🌟 **[2025-11-07]** [EASI v0.1.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.0) is released. Major updates include:
- Supports 7 recent Spatial Intelligence models:
    - SenseNova-SI Family: [SenseNova-SI-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-InternVL3-8B), [SenseNova-SI-InternVL3-2B](https://huggingface.co/collections/sensenova/sensenova-si)
    - MindCube Family: [MindCube-3B-RawQA-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT), [MindCube-3B-Aug-CGMap-FFR-Out-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT),[MindCube-3B-Plain-CGMap-FFR-Out-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT)
    - SpatialLadder: [SpatialLadder-3B](https://huggingface.co/hongxingli/SpatialLadder-3B)
    - SpatialMLLM: [SpatialMLLM-4B](https://diankun-wu.github.io/Spatial-MLLM/)
- Supports 6 recent Spatial Intelligence benchmarks:
    - 4 image-based benchmarks: [MindCube](https://mind-cube.github.io/), [ViewSpatial](https://zju-real.github.io/ViewSpatial-Page/), [EmbSpatial](https://github.com/mengfeidu/EmbSpatial-Bench) and [MMSI(no circular evaluation)](https://arxiv.org/abs/2505.23764)
    - 2 image-and-video benchmarks: [VSI-Bench](https://vision-x-nyu.github.io/thinking-in-space.github.io/) and [SITE-Bench](https://wenqi-wang20.github.io/SITE-Bench.github.io/)
- Introduces a standardized testing protocol as outlined in [EASI](https://arxiv.org/pdf/2508.13142)


## 🛠️ QuickStart
### Installation
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./VLMEvalKit
```

### Configuration
**VLM Configuration**: All VLMs are configured in `vlmeval/config.py`. During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM. Make sure you can successfully infer with the VLM before starting the evaluation with the following command `vlmutil check {MODEL_NAME}`.

**Benchmark Configuration**: The full list of supported Benchmarks can be found in the official VLMEvalKit documentation [VLMEvalKit Supported Benchmarks (Feishu)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY). For the [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard), the following Benchmarks are currently supported:
| Benchmark   | Evaluation settings          |
|-------------|------------------------------|
| [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | [VSI-Bench_origin_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench.tsv)  |
|             |  [VSI-Bench-Debiased_origin_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench-Debiased.tsv)             |
| [SITE-Bench](https://huggingface.co/datasets/franky-veteran/SITE-Bench)  | [SiteBenchImage](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchImage.tsv)        |
|             |  [SiteBenchVideo_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchVideo.tsv)             |
| [MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)  | [MMSIBench_wo_circular](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIBench_wo_circular.tsv)        |
| [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube)    | [MindCubeBench_tiny_raw_qa](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_tiny_raw_qa.tsv)    |
|             | [MindCubeBench_raw_qa](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_raw_qa.tsv)         |
| [ViewSpatial](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) | [ViewSpatialBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/ViewSpatialBench.tsv)            |
| [EmbSpatial](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench)  | [EmbSpatialBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/EmbSpatialBench.tsv)             |


### Evaluation
**General command**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --verbose --reuse
```
See `run.py` for the full list of arguments.

**Example** 

Evaluate `SenseNova-SI-1.1-InternVL3-8B` on `MindCubeBench_tiny_raw_qa`:

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.1-InternVL3-8B \
              --verbose --reuse
```

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