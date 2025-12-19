# Changelog

All notable changes to **EASI** are documented in this file.  

---

## [v0.1.4] – 2025-12-19

### Added
- 4 new benchmarks:
  - SPBench, MMSI-Video-Bench, VSI-SUPER-Recall, VSI-SUPER-Count.

---

## [v0.1.3] – 2025-12-12

### Added
- 3 new image benchmarks:
  - ERQA, RefSpatial-Bench, RoboSpatial-Home
- Docker support improvements:
  - Added a generic EASI runtime Dockerfile  
  - Added model-specific Dockerfiles for Cambrian-S and VLM3R  
---

## [v0.1.2] – 2025-12-08

### Added
- 6 new models:
  - SenseNova-SI 1.1: Qwen2.5-VL-3B / Qwen2.5-VL-7B / Qwen3-VL-8B  
  - SenseNova-SI 1.2: InternVL3-8B  
  - VLM-3R  
  - BAGEL-7B-MoT
- 4 new image benchmarks:
  - STAR-Bench  
  - OmniSpatial  
  - Spatial-Visualization-Benchmark  
  - SPAR-Bench  

### New Feature
- LLM-based answer extraction for selected EASI benchmarks  
  - Enable via: `--judge gpt-4o-1120`  
  - Routes to `gpt-4o-2024-11-20`  

---

## [v0.1.1] – 2025-11-21

### Added
- 9 Spatial Intelligence models (total expanded from 7 → 16):
  - SenseNova-SI: InternVL3-8B / InternVL3-2B  
  - SpaceR-7B  
  - VST-3B-SFT / VST-7B-SFT  
  - Cambrian-S: 0.5B / 1.5B / 3B / 7B  
- New benchmark:
  - VSI-Bench-Debiased  

---

## [v0.1.0] – 2025-11-07

### Initial Public Release

### Added
- 7 Spatial Intelligence models:
  - SenseNova-SI (InternVL3-8B / InternVL3-2B)  
  - MindCube Series (3 variants)  
  - SpatialLadder-3B  
  - SpatialMLLM-4B  
- 6 Spatial Intelligence benchmarks:
  - Image: MindCube, ViewSpatial, EmbSpatial, MMSI  
  - Video: VSI-Bench, SITE-Bench  

### Introduced
- EASI standardized evaluation protocol (as described in the paper)  
- Unified VLMEvalKit-based evaluation pipeline  

---

