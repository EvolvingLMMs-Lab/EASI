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

🌟 **[2025-12-05]**  
[EASI v0.1.2](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.2) 发布。主要更新包括：

- **模型支持扩展**  
  新增 **1 个空间智能模型**，模型总数从 **16 个增加至 17 个**：
    - VLM-3R: [VLM-3R](https://github.com/VITA-Group/VLM-3R)

  新增 **1 个统一理解–生成模型（Unified Understanding–Generation Model）**：
    - BAGEL-7B-MoT: [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)

- **基准测试支持扩展**  
  新增 **4 个图像空间智能基准测试**，基准数量从 **7 个增加至 11 个**：
    - [**STAR-Bench**](https://huggingface.co/datasets/internlm/STAR-Bench)  
    - [**OmniSpatial**](https://huggingface.co/datasets/qizekun/OmniSpatial)  
    - [**Spatial-Visualization-Benchmark**](https://huggingface.co/datasets/PLM-Team/Spatial-Visualization-Benchmark)  
    - [**SPAR-Bench**](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)

- **EASI 基准的 LLM 答案抽取评测**  
  为部分 EASI 基准新增可选的「基于大模型的答案抽取」评测模式。你可以通过指定：
  ```bash
  --judge gpt-4o-1120
  ```
  来启用 OpenAI 评测，内部将路由到 gpt-4o-2024-11-20 进行自动打分。

🌟 **[2025-11-21]**
[EASI v0.1.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.1) 发布。主要更新包括：：

- **模型支持扩展**  
  新增 **9 个空间智能模型**，模型总数从 **7 个增加至 16 个**：
    - **SenseNova-SI 1.1 系列**  
        - [SenseNova-SI-1.1-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-8B)  
        - [SenseNova-SI-1.1-InternVL3-2B](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-2B)
    - SpaceR: [SpaceR-7B](https://huggingface.co/RUBBISHLIKE/SpaceR)kv
    - VST 系列: [VST-3B-SFT](https://huggingface.co/rayruiyang/VST-3B-SFT), [VST-7B-SFT](https://huggingface.co/rayruiyang/VST-7B-SFT)
    - Cambrian-S 系列:  
        [Cambrian-S-0.5B](https://huggingface.co/nyu-visionx/Cambrian-S-0.5B),  
        [Cambrian-S-1.5B](https://huggingface.co/nyu-visionx/Cambrian-S-1.5B),  
        [Cambrian-S-3B](https://huggingface.co/nyu-visionx/Cambrian-S-3B), 
        [Cambrian-S-7B](https://huggingface.co/nyu-visionx/Cambrian-S-7B)

- **基准测试支持扩展**  
  新增 **1 个图像–视频空间智能基准测试**，基准数量从 **6 个增加至 7 个**：
    - [**VSI-Bench-Debiased**](https://vision-x-nyu.github.io/thinking-in-space.github.io/)

---

🌟 [2025-11-07] [EASI v0.1.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.0) 发布。主要更新包括：

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
| [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | [VSI-Bench_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench.tsv)  |
|             |  [VSI-Bench-Debiased_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench-Debiased.tsv)             |
| [SITE-Bench](https://huggingface.co/datasets/franky-veteran/SITE-Bench)  | [SiteBenchImage](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchImage.tsv)        |
|             |  [SiteBenchVideo_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SiteBenchVideo.tsv)             |
| [MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)  | [MMSIBench_wo_circular](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIBench_wo_circular.tsv)        |
| [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube)    | [MindCubeBench_tiny_raw_qa](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_tiny_raw_qa.tsv)    |
|             | [MindCubeBench_raw_qa](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_raw_qa.tsv)         |
| [ViewSpatial](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) | [ViewSpatialBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/ViewSpatialBench.tsv)            |
| [EmbSpatial](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench)  | [EmbSpatialBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/EmbSpatialBench.tsv)             |
| [SparBench](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)  | [SparBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SparBench.tsv)             |
|             |  [SparBench_tiny](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SparBench_tiny.tsv)             |
| [STAR-Bench](https://huggingface.co/datasets/internlm/STAR-Bench)  | [StareBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/StareBench.tsv)             |
|             |  [StareBench_CoT](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/StareBench.tsv)             |
| [Spatial-Visualization-Benchmark](https://huggingface.co/datasets/PLM-Team/Spatial-Visualization-Benchmark)  | [SpatialVizBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SpatialVizBench.tsv)             |
|             |  [SpatialVizBench_CoT](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/SpatialVizBench.tsv)             |
| [OmniSpatial](https://huggingface.co/datasets/qizekun/OmniSpatial)  | [OmniSpatialBench](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OmniSpatialBench.tsv)             |
|             |  [OmniSpatialBench_default](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OmniSpatialBench.tsv)             |
|             |  [OmniSpatialBench_zeroshot_cot](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OmniSpatialBench.tsv)             |
|             |  [OmniSpatialBench_manual_cot](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OmniSpatialBench.tsv)             |
### 评测
**通用命令**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --verbose --reuse --judge extract_matching
```
完整参数说明请参见 run.py

**示例**

在 `MindCubeBench_tiny_raw_qa` 上评测 `SenseNova-SI-1.1-InternVL3-8B`：

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.1-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```

### 提交

将您的评测结果提交到我们的 [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard)：

1. 访问 [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard) 页面。
2. 点击 **🚀 Submit here!** 进入提交表单。
3. 按照页面上的说明填写提交表单，并提交你的结果。

## 🖊️ 引用

```bib
@article{easi2025,
  title={Holistic Evaluation of Multimodal LLMs on Spatial Intelligence},
  author={Cai, Zhongang and Wang, Yubo and Sun, Qingping and Wang, Ruisi and Gu, Chenyang and Yin, Wanqi and Lin, Zhiqian and Yang, Zhitao and Wei, Chen and Shi, Xuanke and Deng, Kewang and Han, Xiaoyang and Chen, Zukai and Li, Jiaqi and Fan, Xiangyu and Deng, Hanming and Lu, Lewei and Li, Bo and Liu, Ziwei and Wang, Quan and Lin, Dahua and Yang, Lei},
  journal={arXiv preprint arXiv:2508.13142},
  year={2025}
}
```