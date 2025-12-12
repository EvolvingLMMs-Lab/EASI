# Supported Models & Benchmarks

This page summarizes all **Spatial Intelligence models** and **benchmarks** currently supported by EASI, together with their corresponding `model` / `dataset` names in VLMEvalKit for easy reproduction and configuration.

## 🧠 Supported Models

> In the command line, please use the values in the **Model** column as the `--model` argument.

| Family / Series   | Model                                       | Type | Link |
|-------------------|---------------------------------------------|------|------|
| SenseNova-SI 1.0  | SenseNova-SI-InternVL3-2B                  | SI   | https://huggingface.co/sensenova/SenseNova-SI-InternVL3-2B |
|                   | SenseNova-SI-InternVL3-8B                  | SI   | https://huggingface.co/sensenova/SenseNova-SI-InternVL3-8B |
| SenseNova-SI 1.1  | SenseNova-SI-1.1-Qwen2.5-VL-3B             | SI   | https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen2.5-VL-3B |
|                   | SenseNova-SI-1.1-Qwen2.5-VL-7B             | SI   | https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen2.5-VL-7B |
|                   | SenseNova-SI-1.1-Qwen3-VL-8B               | SI   | https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen3-VL-8B   |
|                   | SenseNova-SI-1.1-InternVL3-2B              | SI   | https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-2B  |
|                   | SenseNova-SI-1.1-InternVL3-8B              | SI   | https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-8B  |
| SenseNova-SI 1.2  | SenseNova-SI-1.2-InternVL3-8B              | SI   | https://huggingface.co/sensenova/SenseNova-SI-1.2-InternVL3-8B  |
| MindCube          | MindCube-3B-RawQA-SFT                      | SI   | https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT     |
|                   | MindCube-3B-Aug-CGMap-FFR-Out-SFT          | SI   | https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Aug-CGMap-FFR-Out-SFT |
|                   | MindCube-3B-Plain-CGMap-FFR-Out-SFT        | SI   | https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-Plain-CGMap-FFR-Out-SFT |
| SpatialLadder     | SpatialLadder-3B                           | SI   | https://huggingface.co/hongxingli/SpatialLadder-3B              |
| SpatialMLLM       | SpatialMLLM-4B                             | SI   | https://huggingface.co/Diankun/Spatial-MLLM-subset-sft/         |
| SpaceR            | SpaceR-7B                                  | SI   | https://huggingface.co/RUBBISHLIKE/SpaceR                       |
| VST               | VST-3B-SFT                                 | SI   | https://huggingface.co/rayruiyang/VST-3B-SFT                    |
|                   | VST-7B-SFT                                 | SI   | https://huggingface.co/rayruiyang/VST-7B-SFT                    |
| Cambrian-S        | Cambrian-S-0.5B                            | SI   | https://huggingface.co/nyu-visionx/Cambrian-S-0.5B              |
|                   | Cambrian-S-1.5B                            | SI   | https://huggingface.co/nyu-visionx/Cambrian-S-1.5B              |
|                   | Cambrian-S-3B                              | SI   | https://huggingface.co/nyu-visionx/Cambrian-S-3B                |
|                   | Cambrian-S-7B                              | SI   | https://huggingface.co/nyu-visionx/Cambrian-S-7B                |
| VLM-3R            | VLM-3R                                     | SI   | https://github.com/VITA-Group/VLM-3R                            |
| BAGEL-7B-MoT      | BAGEL-7B-MoT                               | UMM  | https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT              |

---

## 📊 Supported Benchmarks

> In the command line, please use the values in the **Data** column as the `--data` argument.

| Benchmark                        | Type        | Data                                                                                       | Link |
|----------------------------------|------------|--------------------------------------------------------------------------------------------|------|
| MindCube                         | image      | `MindCubeBench_tiny_raw_qa`, `MindCubeBench_raw_qa`                                       | https://huggingface.co/datasets/MLL-Lab/MindCube |
| ViewSpatial                      | image      | `ViewSpatialBench`                                                                         | https://huggingface.co/datasets/lidingm/ViewSpatial-Bench |
| EmbSpatial-Bench                 | image      | `EmbSpatialBench`                                                                          | https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench |
| MMSI-Bench (no circular)         | image      | `MMSIBench_wo_circular`                                                                    | https://huggingface.co/datasets/RunsenXu/MMSI-Bench |
| VSI-Bench                        | video      | `{VSI-Bench, VSI-Bench-Debiased}_{128frame,64frame,32frame,16frame,2fps,1fps}`            | https://huggingface.co/datasets/nyu-visionx/VSI-Bench |
| SITE-Bench                       | image+video| image: `SiteBenchImage`<br>video: `{SiteBenchVideo}_{64frame,32frame,1fps}`               | https://huggingface.co/datasets/franky-veteran/SITE-Bench |
| SPAR-Bench                       | image      | `SparBench`, `SparBench_tiny`                                                             | https://huggingface.co/datasets/jasonzhango/SPAR-Bench |
| STAR-Bench                       | image      | `StareBench`, `StareBench_CoT`                                                            | https://huggingface.co/datasets/internlm/STAR-Bench |
| Spatial-Visualization-Benchmark  | image      | `SpatialVizBench`, `SpatialVizBench_CoT`                                                  | https://huggingface.co/datasets/PLM-Team/Spatial-Visualization-Benchmark |
| OmniSpatial                      | image      | `OmniSpatialBench`, `OmniSpatialBench_default`, `OmniSpatialBench_zeroshot_cot`, `OmniSpatialBench_manual_cot` | https://huggingface.co/datasets/qizekun/OmniSpatial |
| ERQA                             | image      | `ERQA`                                                                                    | https://huggingface.co/datasets/FlagEval/ERQA |
| RefSpatial-Bench                 | image      | `RefSpatial`, `RefSpatial_wo_unseen`                                                      | https://huggingface.co/datasets/BAAI/RefSpatial-Bench |
| RoboSpatial-Home                 | image      | `RoboSpatialHome`                                                 | https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home |