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

完整的支持模型与基准列表见  👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**。

## 🗓️ 最新动态

🌟 **[2025-12-12]** [EASI v0.1.3](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.3) 发布。主要更新包括：
- **基准测试支持扩展**  
  新增 **3 个图像空间智能基准**: ERQA, RefSpatial-Bench, RoboSpatial-Home.  

- **环境与部署支持优化**  
  新增通用 EASI Dockerfile，以及面向 Cambrian-S 与 VLM3R 的模型专用 Dockerfile，简化环境配置流程，提升评测的可复现性。
---


🌟 **[2025-12-08]** [EASI v0.1.2](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.2) 发布。主要更新包括：

- **模型支持扩展**  
  新增 **5 个空间智能模型** 和 **1 个统一理解–生成模型**：
  - SenseNova-SI 1.1 系列（Qwen2.5-VL-3B / Qwen2.5-VL-7B / Qwen3-VL-8B）
  - SenseNova-SI 1.2 系列（InternVL3-8B）
  - VLM-3R
  - BAGEL-7B-MoT

- **基准测试支持扩展**  
  新增 **4 个图像空间智能基准**：STAR-Bench、OmniSpatial、Spatial-Visualization-Benchmark、SPAR-Bench。


- **EASI 基准的 LLM 答案抽取评测**  
  为多项 EASI 基准新增可选的「基于大语言模型的答案抽取」评测模式。可以通过：
  ```bash
  --judge gpt-4o-1120
  ```
  来启用 OpenAI 评测，内部将路由到 gpt-4o-2024-11-20 进行自动打分。

🌟 **[2025-11-21]** [EASI v0.1.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.1) 发布。主要更新包括：

- **模型支持扩展**  
  新增 **9 个空间智能模型**（模型总数从 **7 个增加至 16 个**）：
  - SenseNova-SI 1.1 系列（InternVL3-8B / InternVL3-2B）
  - SpaceR-7B
  - VST 系列（VST-3B-SFT / VST-7B-SFT）
  - Cambrian-S 系列（0.5B / 1.5B / 3B / 7B）

- **基准测试支持扩展**  
  新增 **1 个图像–视频空间智能基准测试**：VSI-Bench-Debiased。

---


🌟 **[2025-11-07]** [EASI v0.1.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.0) 发布。主要更新包括：

- **模型支持扩展**  
  支持 **7 个空间智能模型**：
  - SenseNova-SI 系列（InternVL3-8B / InternVL3-2B）
  - MindCube 系列（3B-RawQA-SFT / 3B-Aug-CGMap-FFR-Out-SFT / 3B-Plain-CGMap-FFR-Out-SFT）
  - SpatialLadder-3B
  - SpatialMLLM-4B

- **基准测试支持扩展**  
  支持 **6 个空间智能基准测试**：
  - 4 个图像基准：MindCube、ViewSpatial、EmbSpatial、MMSI（no circular evaluation）
  - 2 个图像–视频基准：VSI-Bench、SITE-Bench

- **标准化测试协议**  
  支持 [EASI 论文](https://arxiv.org/pdf/2508.13142) 中提出的标准化测试协议。

## 🛠️ 快速上手
### 安装
#### 方式一：本地环境

```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
pip install -e ./VLMEvalKit
```

#### 方式二：基于Docker

```bash
bash dockerfiles/EASI/build_runtime_docker.sh

docker run --gpus all -it --rm \
  -v /path/to/your/data:/mnt/data \
  --name easi-runtime \
  vlmevalkit_EASI:latest \
  /bin/bash
```

### 配置

VLM 配置：所有 VLM 都在 vlmeval/config.py 中配置。在评测时，你应当使用该文件中 supported_VLM 指定的模型名称来选择 VLM。开始评测前，请先通过如下命令确认该 VLM 可以成功推理： `vlmutil check {MODEL_NAME}`。

基准（Benchmark）配置：完整的已支持基准列表见 VLMEvalKit 官方文档 [VLMEvalKit Supported Benchmarks](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY)。对于 [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard)，所有 EASI 基准测试及其对应的 --data 名称汇总在 [支持的模型和基准测试](docs/Support_bench_models.md) 中。

以下是 EASI Benchmark 设置的一个最小示例：

| Benchmark   | Evaluation settings          |
|-------------|------------------------------|
| [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | [VSI-Bench_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench.tsv)  |
|             |  [VSI-Bench-Debiased_32frame](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/VSI-Bench-Debiased.tsv)             |
| [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube)    | [MindCubeBench_tiny_raw_qa](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_tiny_raw_qa.tsv)    |

有关 EASI 支持的模型和基准，请参阅[支持的模型和基准](docs/Support_bench_models.md)。

### 评测
**通用命令**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --judge {JUDGE_MODE} --verbose --reuse 
```
完整参数说明请参见 run.py

**示例**

在 `MindCubeBench_tiny_raw_qa` 上评测 `SenseNova-SI-1.2-InternVL3-8B`：

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.2-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```
这将使用正则表达式来提取答案。如果您想使用基于 LLM 的评判系统（例如，在评估 SpatialVizBench_CoT 时），您可以将评判系统切换到 OpenAI：
```
python run.py --data SpatialVizBench_CoT \
              --model {MODEL_NAME} \
              --verbose --reuse --judge gpt-4o-1120
```
注意：要使用 OpenAI 模型，必须设置环境变量 `OPENAI_API_KEY`。

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