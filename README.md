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

🌟 **[2025-11-07]** [EASI v0.1.0]() is released. Major updates include:
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
| [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | VSI-Bench_origin_32frame (Generated during evaluation)   |
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

Evaluate `SenseNova-SI-InternVL3-2B` on `MindCubeBench_tiny_raw_qa`:

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-InternVL3-2B \
              --verbose --reuse
```


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