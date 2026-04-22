#!/usr/bin/env bash
# Debug helper: run one LHPR-VLN mirror-prompt episode end-to-end and
# capture the exact flipped PNGs the builder ships to the LLM. The
# capture happens inside LHPRVLNMirrorSFTPromptBuilder (guarded by the
# MIRROR_DEBUG_DIR env var), so what lands on disk *is* what the model
# saw — no post-processing.
#
# Usage:
#   bash debug_mirror_one_episode.sh <model-path> [split]
#
#   split = unseen_val_filtered_sft_mirror (default, smallest split)
#         = unseen_test_filtered_sft_mirror
#
# Env overrides:
#   SIM_GPUS (default: 0)        — habitat_sim render GPU(s)
#   LLM_GPUS (default: 1,2)      — vLLM GPU(s)
#   TP       (default: 2)        — tensor_parallel_size
#
# After the run:
#   debug_logs/<task>/<ts>_<model>/episodes/000_ep_0/step_*.png
#       original sim-rendered front/left/right frames.
#   debug_logs/<task>/<ts>_<model>/episodes/000_ep_0/mirror/step_*.png
#       flipped + slot-swapped frames the mirror builder served to the LLM.
#   debug_logs/<task>/<ts>_<model>/episodes/000_ep_0/mirror/step_NNNN_history_*.png
#       historical front-views the builder sampled (also flipped).

set -euo pipefail

MODEL="${1:-}"
SPLIT="${2:-unseen_val_filtered_sft_mirror}"
TASK="lhpr_vln_${SPLIT}"
OUTPUT_DIR="./debug_logs"
SIM_GPUS="${SIM_GPUS:-0}"
LLM_GPUS="${LLM_GPUS:-1,2}"
TP="${TP:-2}"

if [ -z "$MODEL" ]; then
    echo "Error: model path required."
    echo "Usage: $0 <model-path> [split]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

# Staging dir for the builder's per-step dumps. Uses a timestamp so repeat
# runs don't collide. Moved into the final episode dir after easi finishes.
TS="$(date +%Y%m%d_%H%M%S)"
MIRROR_STAGE="$REPO_ROOT/$OUTPUT_DIR/_mirror_stage_${TS}"
mkdir -p "$MIRROR_STAGE"
export MIRROR_DEBUG_DIR="$MIRROR_STAGE"

echo "=== Mirror debug run ==="
echo "  model:    $MODEL"
echo "  task:     $TASK"
echo "  staging:  $MIRROR_STAGE"
echo "  sim_gpus: $SIM_GPUS   llm_gpus: $LLM_GPUS   tp: $TP"
echo ""

easi start "$TASK" \
    --agent react --backend vllm \
    --model "$MODEL" \
    --episodes :1 \
    --num-parallel 1 \
    --sim-gpus "$SIM_GPUS" \
    --llm-gpus "$LLM_GPUS" \
    --llm-kwargs "{\"tensor_parallel_size\": $TP, \"trust_remote_code\": true, \"startup_timeout\": 900, \"skip_special_tokens\": false}" \
    --output-dir "$OUTPUT_DIR" \
    --verbosity TRACE

RUN_DIR="$(ls -td "$OUTPUT_DIR/$TASK"/*/ | head -1)"
EP_DIR="$(ls -d "$RUN_DIR"episodes/*/ | head -1)"

# Move the staged mirror PNGs into the episode dir so they sit next to the
# originals for easy A/B.
MIRROR_DIR="${EP_DIR}mirror"
mkdir -p "$MIRROR_DIR"
if compgen -G "$MIRROR_STAGE/*.png" > /dev/null; then
    mv "$MIRROR_STAGE"/*.png "$MIRROR_DIR/"
fi
rmdir "$MIRROR_STAGE" 2>/dev/null || true

FRAME_COUNT="$(find "$MIRROR_DIR" -maxdepth 1 -name '*.png' | wc -l)"

echo ""
echo "=== Done ==="
echo "  run dir:    $RUN_DIR"
echo "  originals:  ${EP_DIR}step_*.png"
echo "  mirror:     ${MIRROR_DIR}/  ($FRAME_COUNT frames)"
echo ""
echo "Sanity checks:"
echo "  jq . '${EP_DIR}result.json'"
echo "  head '${EP_DIR}trajectory.jsonl' | jq ."
echo ""
echo "Look for a step where llm_response contains <|left|> and action is"
echo "turn_right (or vice versa). That confirms the remap fired."
