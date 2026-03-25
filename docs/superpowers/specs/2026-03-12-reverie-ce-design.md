# REVERIE-CE Integration Design

**Date:** 2026-03-12
**Status:** Approved

## Overview

Integrate REVERIE-CE (navigation-only) as a new task in EASI, reusing the existing `habitat_sim:v0_1_7` simulator infrastructure and VLN-CE R2R vendor code. REVERIE-CE uses the pre-converted Dynam3D dataset (HuggingFace) with Matterport3D scenes.

## Scope

- **Navigation-only**: No object grounding. The agent navigates to the described area and calls stop.
- **Reuse existing simulator**: Habitat-Sim 0.1.7 (Python 3.8), same as VLN-CE R2R.
- **Pre-converted data**: Use Dynam3D's pre-converted REVERIE-CE episodes from HuggingFace, repackaged into an EASI-compatible repo.

## Architecture

```
easi/tasks/reverie_ce/
├── task.py              # ReverieCETask (inherits from VLNCETask)
├── _base.yaml           # Config pointing to REVERIE-CE dataset repo
├── bridge.py            # ReverieCEBridge (inherits from VLNCEBridge)
├── prompts.py           # ReverieCEPromptBuilder (high-level instruction style)
├── actions.py           # Same 4 discrete actions as VLN-CE R2R
├── vendor/              # Reuse vlnce_r2r vendor code (import, not symlink)
├── reverie_ce_val_unseen.yaml
└── reverie_ce_test.yaml
```

### Key decisions

- **Bridge**: Inherit from `VLNCEBridge` (not symlink) to allow future customization.
- **Task**: Inherit from `VLNCETask` to reuse metric extraction and aggregation.
- **Vendor**: Import `SceneSimulator` and `scene_config` from `vlnce_r2r/vendor/` — no duplication.
- **Simulator**: No new simulator code. Uses `habitat_sim:v0_1_7` as-is.

## Data Pipeline

### Source

Dynam3D pre-converted REVERIE-CE data from HuggingFace (`MrZihanWang/Dynam3D`):
- `reverie_training_data/` — per-scene JSON files (~60 files)
- `reverie_val_unseen_data.json` — single file
- `reverie_test_data.json` — single file
- `reverie_val_unseen_gt.json`, `reverie_test_gt.json` — ground truth

### EASI HuggingFace Repo

Reformat (not re-convert) the Dynam3D output into EASI's per-split JSONL convention and upload to `oscarqjh/REVERIE-CE_easi`. This is a one-time reshaping step — no discrete-to-CE conversion is involved.

```
oscarqjh/REVERIE-CE_easi/
├── data/
│   ├── train.jsonl
│   ├── val_unseen.jsonl
│   ├── test.jsonl
│   ├── val_unseen_gt.json
│   └── test_gt.json
├── mp3d_scenes.zip         # Matterport3D .glb files (shared with R2R)
```

### Episode Format (JSONL)

Each line matches the VLN-CE R2R format used by `vlnce_r2r`:

```json
{
  "episode_id": "50001",
  "scene_id": "mp3d/cV4RVeZvu5T/cV4RVeZvu5T.glb",
  "instruction": "Go to the laundry room and bring me the blue cushion",
  "start_position": [x, y, z],
  "start_rotation": [qx, qy, qz, qw],
  "goal_position": [x, y, z],
  "geodesic_distance": 10.5,
  "gt_locations": [[x1, y1, z1], [x2, y2, z2], ...]
}
```

The `mp3d_scenes.zip` contains the same Matterport3D scenes as R2R. Can be shared or symlinked to avoid duplication on disk.

## Task Configuration

### _base.yaml

```yaml
name: reverie_ce
simulator: "habitat_sim:v0_1_7"
task_class: "easi.tasks.reverie_ce.task.ReverieCETask"

dataset:
  source: huggingface
  repo_id: "oscarqjh/REVERIE-CE_easi"
  hf_data_dir: "data"
  zip_files: ["mp3d_scenes.zip"]

simulator_configs:
  render_platform: auto
  screen_height: 480
  screen_width: 480
  hfov: 90
  sensor_height: 1.25
  forward_step_size: 0.25
  turn_angle: 15
  allow_sliding: true
  gpu_device_id: 0
  success_distance: 3.0
  additional_deps: ["fastdtw>=0.3.4"]

agent:
  prompt_builder: "easi.tasks.reverie_ce.prompts.ReverieCEPromptBuilder"
  prompt_builder_kwargs:
    use_feedback: true
    use_geo_distance: true
    action_history_len: 20
    chat_history: false
    message_window_len: 5
  generation_kwargs:
    temperature: 0
    max_tokens: 4096
    top_p: 0.95
```

### Split configs

Each split YAML extends `_base.yaml`:

```yaml
# reverie_ce_val_unseen.yaml
extends: _base.yaml
name: reverie_ce_val_unseen
dataset:
  split: "val_unseen"
```

```yaml
# reverie_ce_test.yaml
extends: _base.yaml
name: reverie_ce_test
dataset:
  split: "test"
```

## Action Space

Same 4 discrete actions as VLN-CE R2R:

| Action | Effect |
|---|---|
| `move_forward` | Move 0.25m forward |
| `turn_left` | Turn 15 degrees left |
| `turn_right` | Turn 15 degrees right |
| `stop` | End navigation, evaluate success |

## Metrics

Same as VLN-CE R2R (navigation-only):

| Metric | Description |
|---|---|
| SR (Success Rate) | 1.0 if agent stops within 3m of goal |
| SPL | Success weighted by path efficiency |
| NE (Navigation Error) | Geodesic distance to goal at stop |
| Oracle SR | Best geodesic distance achieved during episode |
| NDTW | Normalized Dynamic Time Warping |
| SDTW | Success-weighted DTW |
| path_length | Total distance traveled |
| steps_taken | Number of actions executed |

Implemented by inheriting `VLNCETask.evaluate_episode()` and `aggregate_results()`.

## Prompt Builder

### Differences from VLN-CE R2R

REVERIE uses high-level instructions ("Go to the laundry room and bring me the blue cushion") vs R2R's turn-by-turn route descriptions ("Exit the bedroom and turn left, walk past the kitchen...").

The system prompt should:
- Frame the task as "navigate to the described area" rather than "follow route instructions"
- Emphasize spatial reasoning from the high-level description
- Otherwise keep the same structure: image, instruction, geodesic feedback, action history

### Response format

Same JSON format as VLN-CE R2R:

```json
{
  "visual_state_description": "...",
  "reasoning_and_reflection": "...",
  "language_plan": "...",
  "executable_plan": [{"action": "move_forward"}]
}
```

## Component Reuse Summary

| Component | Source | Method |
|---|---|---|
| Simulator | `habitat_sim:v0_1_7` | As-is, no changes |
| Bridge | `vlnce_r2r.bridge.VLNCEBridge` | Inherit |
| SceneSimulator | `vlnce_r2r.vendor.scene_simulator` | Import |
| scene_config | `vlnce_r2r.vendor.scene_config` | Import |
| DTW metrics | `vlnce_r2r.vendor.dtw` | Import |
| Task class | `vlnce_r2r.task.VLNCETask` | Inherit |
| Actions | `vlnce_r2r.actions` | Import or duplicate (trivial) |
| Prompt builder | New | Adapted for high-level instructions |
| Dataset | New HuggingFace repo | Repackaged from Dynam3D |

## CLI Usage

```bash
# List available tasks
easi task list  # Should show reverie_ce_val_unseen, reverie_ce_test

# Download dataset
easi task download reverie_ce_val_unseen

# Run evaluation
easi start reverie_ce_val_unseen --agent react --backend openai --model gpt-4o

# Parallel evaluation
easi start reverie_ce_val_unseen --agent react --backend openai --model gpt-4o --num-parallel 4
```

## References

- [Dynam3D (GitHub)](https://github.com/MrZihan/Dynam3D) — conversion scripts and pre-converted data
- [Dynam3D (HuggingFace)](https://huggingface.co/datasets/MrZihanWang/Dynam3D) — pre-converted REVERIE-CE episodes
- [VLN-CE (GitHub)](https://github.com/jacobkrantz/VLN-CE) — original R2R-CE implementation
- [REVERIE (GitHub)](https://github.com/YuankaiQi/REVERIE) — original discrete REVERIE benchmark
- [REVE-CE (IEEE Xplore)](https://ieeexplore.ieee.org/document/9674225) — prior work porting REVERIE to CE (no public code)
