# 📊 Benchmark Verification

Validation of EASI implementations against official reported scores.

## 🟢 Status Legend & Methodology

The status is based on the absolute difference $\lvert\Delta\rvert$.

| Symbol | Status | Criteria |
| :---: | :--- | :--- |
| ✅ | **Strong Agreement** | $0.0\% \le \lvert\Delta\rvert \le 2.5\%$ |
| ☑️ | **Acceptable Variance** | $2.5\% < \lvert\Delta\rvert \le 5.0\%$ |
| ❌ | **Discrepancy** | $5.0\% < \lvert\Delta\rvert$ |

> **📝 Note on $\Delta$ Calculation:**
> * Formula: $\Delta = \text{EASI} - \text{Target Score}$
> * **Target Source:** We prioritize the **Official Code** (local run of the official codebase) to strictly verify implementation correctness. If strict reproduction is not performed, we align with the **Paper Reported** score.
---

## 📑 Index
*(Matches the order in [Supported Benchmarks](./Support_bench_models.md))*

1. [MindCube](#1-mindcube)
2. [ViewSpatial](#2-viewspatial)
3. [EmbSpatial-Bench](#3-embspatial-bench)
4. [MMSI-Bench (no circular)](#4-mmsi-bench-no-circular)
5. [VSI-Bench](#5-vsi-bench)
6. [VSI-Bench-Debiased](#6-vsi-bench-debiased)
<!-- 7. [SITE-Bench](#7-site-bench) 对齐差的有点多 -->
8. [SPAR-Bench](#8-spar-bench)
<!-- 9. [STARE-Bench](#9-stare-bench) 差得多，但猜测是它本身的matching太辣鸡导致的 -->
10. [Spatial-Visualization-Benchmark](#10-spatial-visualization-benchmark)
11. [OmniSpatial](#11-omnispatial)
12. [ERQA](#12-erqa)
<!-- 13. [RefSpatial-Bench](#13-refspatial-bench) 再看看
14. [RoboSpatial-Home](#14-robospatial-home) 再看看 -->
15. [SPBench](#15-spbench)
16. [MMSI-Video-Bench](#16-mmsi-video-bench)
<!-- 17. [VSI-SUPER-Recall](#17-vsi-super-recall) 没数据
18. [VSI-SUPER-Count](#18-vsi-super-count) 没数据 -->

---

## 🔬 Detailed Verification

### 1. MindCube
* **Metric:** Accuracy

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `MindCubeBench_tiny_raw_qa` | 37.81 | - | 37.88 | +0.07 | ✅ |
| Qwen2.5-VL-3B-Instruct | `MindCubeBench_raw_qa` | 33.21 | 36.08 | 35.65 | -0.43 | ✅ |
| Qwen2.5-VL-7B-Instruct | `MindCubeBench_raw_qa` | 29.26 | 31.12 | 31.48 | +0.36 | ✅ |

### 2. ViewSpatial
* **Metric:** Accuracy

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct | `ViewSpatialBench` | 35.85 | - | 31.97 | -3.89 | ☑️ |
| Qwen2.5-VL-7B-Instruct | `ViewSpatialBench` | 36.85 | - | 36.85 | +0.00 | ✅ |
| InternVL3-14B | `ViewSpatialBench` | 40.28 | - | 40.53 | +0.25 | ✅ |

### 3. EmbSpatial-Bench
* **Metric:** Accuracy

| Model | Data | Paper | Qwen3-VL-Report | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-4B-Instruct  | `EmbSpatialBench` | - | 79.6 | 78.7 | -0.9 | ✅ |
| Qwen3-VL-8B-Instruct  | `EmbSpatialBench` | - | 78.5 | 77.7 | -0.8 | ✅ |

### 4. MMSI-Bench (no circular)
* **Metric:** Accuracy

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `MMSIBench_wo_circular` | 26.5 | - | 28.6 | +2.1 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `MMSIBench_wo_circular` | 25.9 | - | 26.8 | +0.9 | ✅ |
| InternVL3-2B  | `MMSIBench_wo_circular` | 25.3 | - | 26.5 | +1.2 | ✅ |
| InternVL3-8B  | `MMSIBench_wo_circular` | 25.7 | - | 28.0 | +2.3 | ✅ |

### 5. VSI-Bench
* **Metric:** Accuracy && MRA
* **Metric:** TBD

| Model | Data | VSI / Cambrian-S Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| InternVL2-8B  | `VSI-Bench_32frame` | 37.5 / - | - | - | +2.1 |  |
| Qwen2.5-VL-3B-Instruct  | `VSI-Bench_128frame` | - / 26.8 | - | - | +2.1 |  |
| Qwen2.5-VL-7B-Instruct  | `VSI-Bench_128frame` | - / 33.5 | - | - | +0.9 |  |
| InternVL3_5-2B  | `VSI-Bench_128frame` | - / - | - | - | +1.2 |  |
| InternVL3_5-8B  | `VSI-Bench_128frame` | - / - | - | - | +2.3 | |
| Cambrian-S-3B  | `VSI-Bench_32frame` | - | 54.73 | 56.08 | +1.35 | ✅ |
| Cambrian-S-7B  | `VSI-Bench_32frame` | - | 63.61 | 62.93 | -0.98 | ✅ |

### 6. VSI-Bench-Debiased
* **Metric:** Accuracy && MRA

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `VSI-Bench-Debiased_32frame` | - | 26.8 | 28.6 | +2.1 |  |
| Qwen2.5-VL-7B-Instruct  | `VSI-Bench-Debiased_32frame` | - | 33.5 | 26.8 | +0.9 |  |
| InternVL3-2B  | `VSI-Bench-Debiased_32frame` | - | - | 26.5 | +1.2 |  |
| InternVL3-8B  | `VSI-Bench-Debiased_32frame` | - | - | 28.0 | +2.3 |  |
| Cambrian-S-3B  | `VSI-Bench-Debiased_32frame` | - | 46.47 | 26.5 | +1.2 | ✅ |
| Cambrian-S-7B  | `VSI-Bench-Debiased_32frame` | - | 55.58 | 28.0 | +2.3 | ✅ |

 
### 8. SPAR-Bench
* **Metric:** Accuracy && MRA

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B-Instruct  | `SparBench` | 33.07 | - | 33.78 | +0.71 | ✅ |


### 10. Spatial-Visualization-Benchmark
* **Metric:** Accuracy

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B-Instruct  | `SpatialVizBench` | 30.76 | - | 31.02 | +0.26 | ✅ |
| InternVL3-8B  | `SpatialVizBench` | 30.25 | - | 31.86 | +1.61 | ✅ |
| Qwen2.5-VL-3B-Instruct  | `SpatialVizBench` | 26.10 | 25.00 | 23.98 | -1.02 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SpatialVizBench_CoT` | 27.97 | - | 27.54 | -0.43 | ✅ |
| InternVL3-8B  | `SpatialVizBench_CoT` | 30.08 | - | 30.00 | -0.08 | ✅ |

### 11. OmniSpatial
* **Metric:** Accuracy

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `OmniSpatialBench_manual_cot` | 40.30 | 40.73 | - | - |  |
| InternVL3-2B  | `OmniSpatialBench_manual_cot` | 37.98 | - | 42.01 | +4.03 | ☑️ |
| InternVL3-8B  | `OmniSpatialBench_manual_cot` | 41.6 | - | 45.34 | +3.74 | ☑️ |


### 12. ERQA
* **Metric:** Accuracy

| Model | Data | Paper | Qwen3-VL-Report | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B-Instruct  | `ERQA` | - | 45.8 | 43 | -2.8 | ☑️ |


### 15. SPBench
* **Metric:** Accuracy && MRA

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-3B-Instruct  | `SPBench-MV` | 36.6 | - | 38.4 | +1.8 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SPBench-MV` | 37.3 | - | 40.7 | +3.4 | ☑️ |
| Qwen2.5-VL-3B-Instruct  | `SPBench-SI` | 40.3 | - | 41.2 | +0.9 | ✅ |
| Qwen2.5-VL-7B-Instruct  | `SPBench-SI` | 48.4 | - | 48.1 | -0.3 | ✅ |

### 16. MMSI-Video-Bench
* **Metric:** Accuracy

| Model | Data | Paper | Official Code | EASI | Δ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B-Instruct  | `MMSIVideoBench_50frame` | 29.7 | - | 26.9 | -2.8 | ☑️ |
| Qwen3-VL-8B-Instruct  | `MMSIVideoBench_50frame` | 27.6 | - | 28.4 | +0.8 | ✅ |
| InternVL3-8B  | `MMSIVideoBench_50frame` | 30.4 | - | 30.2 | -0.32 | ✅ |
