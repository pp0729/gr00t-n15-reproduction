#!/usr/bin/env bash
set -euo pipefail

# Intended to run on the server after Git synchronization.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PYTHON="/mnt/4t/guo/miniconda3/envs/gr00t-n15/bin/python"
BASE_MODEL="/mnt/4t/guo/cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
DATASET="$PROJECT_ROOT/demo_data/robot_sim.PickNPlace"

cd "$PROJECT_ROOT"

[[ -x "$TRAIN_PYTHON" ]] || {
    echo "Training Python not found: $TRAIN_PYTHON" >&2
    exit 1
}
[[ -d "$BASE_MODEL" && -d "$DATASET" ]] || {
    echo "Base model or dataset directory missing." >&2
    exit 1
}

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is not clean. Review changes before training." >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="/mnt/4t/guo/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

# Each execution creates a separate output directory.
EXPERIMENT_ROOT="/mnt/4t/guo/gr00t-experiments/finetune-smoke"
mkdir -p "$EXPERIMENT_ROOT"
RUN_DIR="$(mktemp -d "$EXPERIMENT_ROOT/$(date +%Y%m%d-%H%M%S)-XXXXXX")"

COMMAND=(
    "$TRAIN_PYTHON" -B -u scripts/gr00t_finetune.py
    --dataset-path "$DATASET"
    --output-dir "$RUN_DIR/checkpoints"
    --base-model-path "$BASE_MODEL"
    --data-config fourier_gr1_arms_only
    --embodiment-tag gr1
    --num-gpus 1
    --batch-size 2
    --max-steps 10
    --save-steps 10
    --dataloader-num-workers 2
    --dataloader-prefetch-factor 2
    --gradient-accumulation-steps 1
    --video-backend torchvision_av
    --report-to tensorboard
    --no-tune-llm
    --no-tune-visual
    --tune-projector
    --tune-diffusion-model
    --no-resume
)

{
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'project_root=%s\n' "$PROJECT_ROOT"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'python=%s\n' "$TRAIN_PYTHON"
    printf 'base_model=%s\n' "$BASE_MODEL"
    printf 'dataset=%s\n' "$DATASET"
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "$RUN_DIR/run-info.txt"

echo "Run directory: $RUN_DIR"

# Record and verify which source tree will actually be imported.
"$TRAIN_PYTHON" -B -c \
    'import os, pathlib, gr00t; p=pathlib.Path(gr00t.__file__).resolve(); print("GR00T source:", p); assert p.parent.parent == pathlib.Path(os.environ["PYTHONPATH"]).resolve()' \
    2>&1 | tee "$RUN_DIR/source-check.log"

nvidia-smi > "$RUN_DIR/gpu-before.txt"

# Preserve the training exit code even though output is piped to tee.
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/train.log"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e

TRAIN_EXIT="${PIPE_RESULTS[0]}"
LOG_EXIT="${PIPE_RESULTS[1]}"

printf 'training_exit_code=%s\nlog_exit_code=%s\n' \
    "$TRAIN_EXIT" "$LOG_EXIT" > "$RUN_DIR/exit-status.txt"

echo "Training exit code: $TRAIN_EXIT"
echo "Results: $RUN_DIR"

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
    exit "$TRAIN_EXIT"
fi
exit "$LOG_EXIT"