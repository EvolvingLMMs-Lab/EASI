<div align="center">
  <img src="assets/banner.jpg"/>

  <b>EASI: Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</b>

  [English](README.md) | 简体中文

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

## 快速了解（TL;DR）

- EASI 是一个面向多模态大模型空间智能（Spatial Intelligence）的统一评测套件。
- EASI 支持**两种评测后端**：[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 和 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)。
- 完成安装后，可以用一条命令运行所有 EASI-8 基准测试：

**使用 VLMEvalKit 后端（默认）：**
```bash
python scripts/submissions/run_easi_eval.py \
  --model sensenova/SenseNova-SI-1.3-InternVL3-8B \
  --nproc 4
```

**使用 lmms-eval 后端：**
```bash
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model internvl2 \
  --model-args "pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B" \
  --nproc 4
```

> EASI 底层封装了 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 和 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)，提供统一的命令行接口。如需高级用法或添加自定义模型，请参阅各后端仓库。

## 概述

EASI 是一个面向空间智能的统一评测套件，用于在不断扩展的空间基准上评估最先进的闭源和开源多模态大模型。

- **广泛支持**：目前 EASI([v0.2.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.1))支持 **23 个空间智能模型**和 **27 个空间基准测试**。
- **双后端支持**：
  - **VLMEvalKit**：丰富的模型库，内置评判功能。
  - **lmms-eval**：轻量级、基于 accelerate 的分布式评测，支持大量任务。

更多详情请参阅 👉 **[Supported Models & Benchmarks](docs/Support_bench_models.md)**。EASI 同时提供透明的 👉 **[Benchmark Verification](docs/Benchmark_Verification.md)** 以供与官方分数对比。

## 🗓️ 最新动态

🌟 **[2026-02-09]** [EASI v0.2.1](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.1) 发布。主要更新包括：
- **新增基准支持**：新增 ERIQ 和 OSI-Bench。
- **Bug 修复**：修复 VLMEvalKit 在 MuirBench 上的评测问题。
- **基准验证**：补充了更多 lmms-eval 的基准验证信息。

🌟 **[2026-01-16]** [EASI v0.2.0](https://github.com/EvolvingLMMs-Lab/EASI/releases/tag/0.2.0) 发布。主要更新包括：
- **新增后端支持**：集成了 lmms-eval 与 VLMEvalKit，提供灵活的评测选择。
- **基准支持扩展**：新增 DSR-Bench。

完整发版历史和详细更新日志，请参见 👉 **[Changelog](docs/CHANGELOG.md)**。

## 🛠️ 快速上手
### 安装

安装脚本会自动安装两种评测后端（VLMEvalKit 和 lmms-eval）及所有依赖：

#### 方式一：本地环境（推荐）

```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI
bash scripts/setup.sh
source .venv/bin/activate
```

该脚本创建 Python 3.11 虚拟环境，安装两个后端、flash-attn 及所有必需依赖。详见 [scripts/setup.sh](scripts/setup.sh)。

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

EASI 提供统一评测脚本，支持 VLMEvalKit 和 lmms-eval 两种后端。脚本自动处理数据集准备、评测、结果收集和排行榜提交。

---

#### 使用统一评测脚本（推荐）

**VLMEvalKit 后端（默认）：**
```bash
# 使用 4 个 GPU 运行 EASI-8 核心基准测试
python scripts/submissions/run_easi_eval.py \
  --model sensenova/SenseNova-SI-1.3-InternVL3-8B \
  --nproc 4
```

**lmms-eval 后端：**
```bash
# 使用 4 个 GPU 运行 EASI-8 核心基准测试
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model internvl2 \
  --model-args "pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B" \
  --nproc 4
```

**带自动提交：**
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

**更多选项：**
```bash
# 运行特定基准
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --benchmarks vsi_bench,blink,sitebench

# 包含额外基准（MMSI-Video, OmniSpatial, SPAR-Bench, VSI-Debiased）
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 --include-extra

# 强制重新评测（忽略已有结果）
python scripts/submissions/run_easi_eval.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --nproc 8 --rerun

# lmms-eval OOM 恢复：用单 GPU 模式完成失败的基准
python scripts/submissions/run_easi_eval.py \
  --backend lmms-eval \
  --model qwen3_vl \
  --model-args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=flash_attention_2" \
  --no-accelerate
```

完整 CLI 选项和提交配置说明请见 👉 **[Submission Guide](docs/Submit_results.md)**。

---

#### 直接使用后端

如需高级用法或添加自定义模型，也可以直接调用后端：

**VLMEvalKit：**
```bash
cd VLMEvalKit/
python run.py --data MindCubeBench_tiny_raw_qa \
              --model SenseNova-SI-1.3-InternVL3-8B \
              --verbose --reuse --judge extract_matching
```

**lmms-eval：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 -m lmms_eval \
    --model internvl2 \
    --model_args=pretrained=sensenova/SenseNova-SI-1.3-InternVL3-8B \
    --tasks vsibench_multiimage \
    --batch_size 1 --log_samples --output_path ./logs/
```

更多后端详情请参阅 [VLMEvalKit 文档](VLMEvalKit/README.md) 和 [lmms-eval 文档](lmms-eval/README.md)。

---

### 配置

- **支持的模型与基准**：汇总于 [支持的模型与基准](docs/Support_bench_models.md)。
- **VLMEvalKit 模型**：定义在 `vlmeval/config.py` 中。请使用 `vlmutil check {MODEL_NAME}` 验证推理是否可用。
- **lmms-eval 模型**：支持多种模型类型（如 `qwen2_5_vl`, `llava`, `internvl2` 等）。详见 [lmms-eval 模型目录](lmms-eval/lmms_eval/models/)。

### 提交

你可以通过 👉 **[EASI Leaderboard Submission](https://easi.lmms-lab.com/submit/)** 提交评测结果。

详细提交要求和文件示例请见 👉 **[Submission Guide](docs/Submit_results.md)**。

## 🤝 贡献

EASI 是一个开放且持续发展的评测套件。我们热忱欢迎社区贡献，包括：
- 新的空间基准测试
- 新的模型基线
- 评测工具

如果您有兴趣参与贡献，或有关于集成的问题，请联系我们：
📧 easi-lmms-lab@outlook.com

## 🖊️ 引用

```bib
@article{easi2025,
  title={Holistic Evaluation of Multimodal LLMs on Spatial Intelligence},
  author={Cai, Zhongang and Wang, Yubo and Sun, Qingping and Wang, Ruisi and Gu, Chenyang and Yin, Wanqi and Lin, Zhiqian and Yang, Zhitao and Wei, Chen and Shi, Xuanke and Deng, Kewang and Han, Xiaoyang and Chen, Zukai and Li, Jiaqi and Fan, Xiangyu and Deng, Hanming and Lu, Lewei and Li, Bo and Liu, Ziwei and Wang, Quan and Lin, Dahua and Yang, Lei},
  journal={arXiv preprint arXiv:2508.13142},
  year={2025}
}
```
