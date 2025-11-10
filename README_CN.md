# EASI

<b>Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</b>

[English](README.md) | 简体中文

<p align="center">
    <a href="https://arxiv.org/abs/2508.13142" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-EASI-red?logo=arxiv" height="20" />
    </a>
    <a href="https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard" target="_blank">
        <img alt="Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_EASI-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/EvolvingLMMs-Lab/EASI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/EvolvingLMMs-Lab/EASI?style=flat"></a>
</p>

## 概述

EASI 构建了一个全面的空间任务分类体系，制定了一套标准化评测协议。EASI统一了近期提出的多项空间智能基准测试，用于对当前最先进的闭源模型和开源模型进行公平评估。

主要特点包括：

- 支持评估**最先进的空间智能模型**。
- 系统性地收集和整合**不断演进的空间智能基准测试**。
- 提出**标准化测试协议**，确保公平评估并支持跨基准测试的比较。

## 🗓️ 最新动态
🌟 [2025-11-07] [EASI v0.1.0]() 发布。主要更新包括：

- 支持 7 个最新的空间智能模型：
    - SenseNova-SI系列: [SenseNova-SI-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-InternVL3-8B), [SenseNova-SI-InternVL3-2B](https://huggingface.co/collections/sensenova/sensenova-si)
    - MindCube系列: [MindCube-3B-RawQA-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT), [MindCube-3B-Aug-CGMap-FFR-Out-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT),[MindCube-3B-Plain-CGMap-FFR-Out-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT)
    - SpatialLadder: [SpatialLadder-3B](https://huggingface.co/hongxingli/SpatialLadder-3B)
    - SpatialMLLM: [SpatialMLLM-4B](https://diankun-wu.github.io/Spatial-MLLM/)
- 支持 6 个最近的空间智能基准测试：
    - 4个基于图像的空间智能基准测试: [MindCube](https://mind-cube.github.io/), [ViewSpatial](https://zju-real.github.io/ViewSpatial-Page/), [EmbSpatial](https://github.com/mengfeidu/EmbSpatial-Bench) and [MMSI(no circular evaluation)](https://arxiv.org/abs/2505.23764)
    - 2个基于图像和视频的空间智能基准测试: [VSI-Bench](https://vision-x-nyu.github.io/thinking-in-space.github.io/) and [SITE-Bench](https://wenqi-wang20.github.io/SITE-Bench.github.io/)
- 支持[EASI](https://arxiv.org/pdf/2508.13142)中提出的标准化测试协议

## 🛠️ 快速上手
### 安装
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./VLMEvalKit
```

### 配置

VLM 配置：所有 VLM 都在 vlmeval/config.py 中配置。在评测时，你应当使用该文件中 supported_VLM 指定的模型名称来选择 VLM。开始评测前，请先通过如下命令确认该 VLM 可以成功推理：vlmutil check {MODEL_NAME}。

基准（Benchmark）配置：完整的已支持基准列表见 VLMEvalKit 官方文档 [VLMEvalKit Supported Benchmarks (Feishu)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY)。对于 [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard)，当前支持的基准如下：

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

### 评测
**通用命令**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --verbose --reuse
```
完整参数说明请参见 run.py

**示例**

在 `MindCubeBench_tiny_raw_qa` 上评测 `SenseNova-SI-InternVL3-2B`：

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-InternVL3-2B \
              --verbose --reuse
```

## 🖊️ 引用

空间智能是一个快速发展的领域，我们的评测范围已经从GPT-5拓展到更多的模型，因此论文的标题改为了[*Holistic Evaluation of Multimodal LLMs on Spatial Intelligence*](https://arxiv.org/abs/2508.13142)。但是为了方便大家引用，BibTeX中还是使用最初的标题。

```bib
@article{easi2025,
  title={Has gpt-5 achieved spatial intelligence? an empirical study},
  author={Cai, Zhongang and Wang, Yubo and Sun, Qingping and Wang, Ruisi and Gu, Chenyang and Yin, Wanqi and Lin, Zhiqian and Yang, Zhitao and Wei, Chen and Shi, Xuanke and Deng, Kewang and Han, Xiaoyang and Chen, Zukai and Li, Jiaqi and Fan, Xiangyu and Deng, Hanming and Lu, Lewei and Li, Bo and Liu, Ziwei and Wang, Quan and Lin, Dahua and Yang, Lei},
  journal={arXiv preprint arXiv:2508.13142},
  year={2025}
}
```