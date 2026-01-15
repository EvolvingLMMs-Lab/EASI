# Supported Models & Benchmarks

This page summarizes all **Spatial Intelligence models** and **benchmarks** currently supported by EASI, together with their corresponding `model` / `dataset` names for easy reproduction and configuration.

## 🧠 Supported Models

> **VLMEvalKit**: Use `--model <Model Name>` (e.g., `--model Qwen2.5-VL-3B-Instruct`)
>
> **LMMs-Eval**: Use `--model <Base Model> --model_args pretrained=<Link>` (e.g., `--model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct`)

| Family | Type | Base Model | Model Name | Link | VLMEvalKit | LMMs-Eval |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **SenseNova-SI 1.0** | SI | `internvl3` | `SenseNova-SI-InternVL3-2B` | [sensenova/SenseNova-SI-InternVL3-2B](https://huggingface.co/sensenova/SenseNova-SI-InternVL3-2B) | ✅ | ✅ |
| | | `internvl3` | `SenseNova-SI-InternVL3-8B` | [sensenova/SenseNova-SI-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-InternVL3-8B) | ✅ | ✅ |
| **SenseNova-SI 1.1** | SI | `qwen2_5_vl` | `SenseNova-SI-1.1-Qwen2.5-VL-3B` | [sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B](https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B) | ✅ | ✅ |
| | | `qwen2_5_vl` | `SenseNova-SI-1.1-Qwen2.5-VL-7B` | [sensenova/SenseNova-SI-1.1-Qwen2.5-VL-7B](https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen2.5-VL-7B) | ✅ | ✅ |
| | | `qwen3_vl` | `SenseNova-SI-1.1-Qwen3-VL-8B` | [sensenova/SenseNova-SI-1.1-Qwen3-VL-8B](https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen3-VL-8B) | ✅ | ✅ |
| | | `internvl3` | `SenseNova-SI-1.1-InternVL3-2B` | [sensenova/SenseNova-SI-1.1-InternVL3-2B](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-2B) | ✅ | ✅ |
| | | `internvl3` | `SenseNova-SI-1.1-InternVL3-8B` | [sensenova/SenseNova-SI-1.1-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-8B) | ✅ | ✅ |
| | UMM | `bagel` | `SenseNova-SI-1.1-BAGEL-7B-MoT` | [sensenova/SenseNova-SI-1.1-BAGEL-7B-MoT](https://huggingface.co/sensenova/SenseNova-SI-1.1-BAGEL-7B-MoT) | ✅ | ✅ |
| **SenseNova-SI 1.2** | SI | `internvl3` | `SenseNova-SI-1.2-InternVL3-8B` | [sensenova/SenseNova-SI-1.2-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-1.2-InternVL3-8B) | ✅ | ✅ |
| **SenseNova-SI 1.3** | SI | `internvl3` | `SenseNova-SI-1.3-InternVL3-8B` | [sensenova/SenseNova-SI-1.3-InternVL3-8B](https://huggingface.co/sensenova/SenseNova-SI-1.3-InternVL3-8B) | ✅ | ✅ |
| **MindCube** | SI | `qwen2_5_vl` | `MindCube-3B-RawQA-SFT` | [MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT) | ✅ | ✅ |
| | | `qwen2_5_vl` | `MindCube-3B-Aug-CGMap-FFR-Out-SFT` | [MLL-Lab/MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT) | ✅ | ✅ |
| | | `qwen2_5_vl` | `MindCube-3B-Plain-CGMap-FFR-Out-SFT` | [MLL-Lab/MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT) | ✅ | ✅ |
| **SpatialLadder** | SI | `qwen2_5_vl` | `SpatialLadder-3B` | [hongxingli/SpatialLadder-3B](https://huggingface.co/hongxingli/SpatialLadder-3B) | ✅ | ✅ |
| **SpatialMLLM** | SI | `internvl2` | `SpatialMLLM-4B` | [Diankun/Spatial-MLLM-subset-sft](https://huggingface.co/Diankun/Spatial-MLLM-subset-sft/) | ✅ | ✅ |
| **SpaceR** | SI | `qwen2_5_vl` | `SpaceR-7B` | [RUBBISHLIKE/SpaceR](https://huggingface.co/RUBBISHLIKE/SpaceR) | ✅ | ✅ |
| **VST** | SI | `qwen2_5_vl` | `VST-3B-SFT` | [rayruiyang/VST-3B-SFT](https://huggingface.co/rayruiyang/VST-3B-SFT) | ✅ | ✅ |
| | | `qwen2_5_vl` | `VST-7B-SFT` | [rayruiyang/VST-7B-SFT](https://huggingface.co/rayruiyang/VST-7B-SFT) | ✅ | ✅ |
| **Cambrian-S** | SI | `cambrians` | `Cambrian-S-0.5B` | [nyu-visionx/Cambrian-S-0.5B](https://huggingface.co/nyu-visionx/Cambrian-S-0.5B) | ✅ | ✅ |
| | | `cambrians` | `Cambrian-S-1.5B` | [nyu-visionx/Cambrian-S-1.5B](https://huggingface.co/nyu-visionx/Cambrian-S-1.5B) | ✅ | ✅ |
| | | `cambrians` | `Cambrian-S-3B` | [nyu-visionx/Cambrian-S-3B](https://huggingface.co/nyu-visionx/Cambrian-S-3B) | ✅ | ✅ |
| | | `cambrians` | `Cambrian-S-7B` | [nyu-visionx/Cambrian-S-7B](https://huggingface.co/nyu-visionx/Cambrian-S-7B) | ✅ | ✅ |
| **VLM-3R** | SI | - | `VLM-3R` | [VITA-Group/VLM-3R](https://github.com/VITA-Group/VLM-3R) | ✅ | ❌ |
| **BAGEL** | UMM | `bagel` | `BAGEL-7B-MoT` | [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | ✅ | ✅ |

---

## 📊 Supported Benchmarks

> **VLMEvalKit**: Use `--data <VLMEvalKit Data>` (e.g., `--data VSI-Bench_32frame`)
>
> **LMMs-Eval**: Use `--tasks <LMMs-Eval Task>` (e.g., `--tasks vsibench`)

| Benchmark | Type | VLMEvalKit Data | LMMs-Eval Task | Release Date |
| :--- | :--- | :--- | :--- | :--- |
| [**MindCube**](https://huggingface.co/datasets/MLL-Lab/MindCube) | image | `MindCubeBench_tiny_raw_qa`,<br>`MindCubeBench_raw_qa` | `mindcube_tiny`,<br>`mindcube_full` | Jun 2025 |
| [**ViewSpatial**](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) | image | `ViewSpatialBench` | `viewspatial` | May 2025 |
| [**EmbSpatial-Bench**](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench) | image | `EmbSpatialBench` | `embspatial` | Jun 2024 |
| [**MMSI-Bench** (no circular)](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) | image | `MMSIBench_wo_circular` | `mmsi_bench` | May 2025 |
| [**VSI-Bench**](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | video | `{VSI-Bench}_`<br>`{128frame,64frame,32frame,16frame,2fps,1fps}` | `vsibench` | Dec 2024 |
| [**VSI-Bench-Debiased**](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | video | `{VSI-Bench-Debiased}_`<br>`{128frame,64frame,32frame,16frame,2fps,1fps}` | `vsibench_debiased` | Nov 2025 |
| [**SITE-Bench**](https://huggingface.co/datasets/franky-veteran/SITE-Bench) | image+video | image: `SiteBenchImage`<br>video: `{SiteBenchVideo}_`<br>`{64frame,32frame,1fps}` | `site_bench_image`,<br>`site_bench_video` | May 2025 |
| [**SPAR-Bench**](https://huggingface.co/datasets/jasonzhango/SPAR-Bench) | image | `SparBench`, `SparBench_tiny` | - | Mar 2025 |
| [**STARE-Bench**](https://huggingface.co/datasets/kuvvi/STARE) | image | `StareBench`, `StareBench_CoT` | `stare_full` | Jun 2025 |
| [**Spatial-Visualization-Benchmark**](https://huggingface.co/datasets/PLM-Team/Spatial-Visualization-Benchmark) | image | `SpatialVizBench`,<br>`SpatialVizBench_CoT` | `spatialviz_full` | Jul 2025 |
| [**OmniSpatial**](https://huggingface.co/datasets/qizekun/OmniSpatial) | image | `OmniSpatialBench`, `OmniSpatialBench_default`,<br>`OmniSpatialBench_zeroshot_cot`,<br>`OmniSpatialBench_manual_cot` | `omnispatial_test` | Jun 2025 |
| [**ERQA**](https://huggingface.co/datasets/FlagEval/ERQA) | image | `ERQA` | `erqa` | Apr 2025 |
| [**RefSpatial-Bench**](https://huggingface.co/datasets/BAAI/RefSpatial-Bench) | image | `RefSpatial`,<br>`RefSpatial_wo_unseen` | `refspatial` | Jun 2025 |
| [**RoboSpatial-Home**](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) | image | `RoboSpatialHome` | - | Nov 2024 |
| [**SPBench**](https://huggingface.co/datasets/hongxingli/SPBench) | image | `SPBench-MV`, `SPBench-SI`,<br>`SPBench-MV_CoT`, `SPBench-SI_CoT` | - | Oct 2025 |
| [**MMSI-Video-Bench**](https://huggingface.co/datasets/rbler/MMSI-Video-Bench) | video | `MMSIVideoBench_`<br>`{300frame,64frame,50frame,32frame,1fps}` | - | Dec 2025 |
| [**VSI-SUPER-Recall**](https://huggingface.co/datasets/nyu-visionx/VSI-SUPER-Recall) | video | `{VsiSuperRecall}_`<br>`{10mins,30mins,60mins,120mins,240mins}_`<br>`{128frame,64frame,32frame,16frame,2fps,1fps}` | - | Nov 2025 |
| [**VSI-SUPER-Count**](https://huggingface.co/datasets/nyu-visionx/VSI-SUPER-Count) | video | `{VsiSuperCount}_`<br>`{10mins,30mins,60mins,120mins}_`<br>`{128frame,64frame,32frame,16frame,2fps,1fps}` | - | Nov 2025 |
| [**STI-Bench**](https://huggingface.co/datasets/MINT-SJTU/STI-Bench) | video | `{STI-Bench}_`<br>`{64frame,32frame,30frame,1fps}` | - | Apr 2025 |
| [**BLINK**](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | image | `BLINK`, `BLINK_circular` | `blink` | Apr 2024 |
| [**CV-Bench**](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | image | `CV-Bench-2D`, `CV-Bench-3D` | `cv_bench`,<br>`cv_bench_2d`, `cv_bench_3d` | Jun 2024 |
| [**3DSRBench**](https://huggingface.co/datasets/ccvl/3DSRBench) | image | `3DSRBench` | - | Dec 2024 |
| [**LEGO-Puzzles**](https://huggingface.co/datasets/KexianTang/LEGO-Puzzles) | image | `LEGO`, `LEGO_circular` | - | Mar 2025 |
| [**Spatial457**](https://huggingface.co/datasets/RyanWW/Spatial457) | image | `Spatial457` | - | Feb 2025 |

