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

## 快速了解（TL;DR）

- EASI 是一个面向多模态大模型空间智能（Spatial Intelligence）的统一评测套件。
- 完成安装后，可以用下面的一行命令快速在 SenseNova-SI 模型上跑一个示例：

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.2-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```

## 概述

EASI 是一个面向空间智能的统一评测套件，用于在不断扩展的空间基准上评估最先进的闭源和开源多模态大模型。

主要特点包括：

- 支持评估**最先进的空间智能模型**。
- 系统性地收集和整合**不断演进的空间智能基准测试**。

在 v0.1.4 版本中，EASI 已支持 **21 个空间智能模型** 和 **17 个空间基准测试**，并将持续扩展。完整的支持模型与基准列表见 👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**。

## 🗓️ 最新动态

🌟 **[2025-12-19]** [EASI v0.1.4](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.1.4) 发布。主要更新包括：
- **基准测试支持扩展**  
  新增 **4 个空间智能基准**：SPBench、MMSI-Video-Bench、VSI-SUPER-Recall、VSI-SUPER-Count。


完整发版历史和详细更新日志，请参见 👉 **[Changelog](docs/CHANGELOG.md)**。

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

### 评测
**通用命令**
```bash
python run.py --data {BENCHMARK_NAME} --model {MODEL_NAME} --judge {JUDGE_MODE} --verbose --reuse 
```
请参阅下方的“配置”部分，查看所有可用模型和基准测试的完整列表。 请参阅 run.py 文件，查看所有参数的完整列表。

**示例**

在 `MindCubeBench_tiny_raw_qa` 上评测 `SenseNova-SI-1.2-InternVL3-8B`：

```bash
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.2-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```
这将使用正则表达式来提取答案。如果您想使用基于 LLM 的评判系统（例如，在评估 SpatialVizBench_CoT 时），您可以将评判系统切换到 OpenAI：
```
export OPENAI_API_KEY=YOUR_KEY
python run.py --data SpatialVizBench_CoT \
              --model {MODEL_NAME} \
              --verbose --reuse --judge gpt-4o-1120
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