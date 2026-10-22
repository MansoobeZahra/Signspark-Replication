#!/usr/bin/env bash
# Train one SignSparK stream with the upstream train.py (unmodified) and the paper's settings.
#   train/run.sh <name> <hand|body> <learning_steps> [extra hydra overrides...]
# Paper settings that differ from the released configs/data_v2/common/train.yaml:
#   loss weights CFM:recon:vel = 2:1:1 (paper Sec. 4.3) and no acceleration loss (not in the paper's Eq. 7);
#   released: lambda_flow 0.5, lambda_recon 1.0, lambda_velocity 0.5, lambda_acceleration 0.1.
#   data = Phoenix14T + CSL-Daily + How2Sign (paper Tab. 7 row 3/4; BOBSL is not released).
# Everything else (UNet-large, M-CLIP pooled text, CFG drop 0.1 for text and keyframes, onset/mid/offset
# keyframes, flip_left_hand, language tag, lr 1e-5 linear anneal, EMA 0.9999, batch 64) is the release default.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME=$1; STREAM=$2; STEPS=$3; shift 3
PY="${PY:-$REPO/../ver-01/.venv-signspark/bin/python}"
OUT="$REPO/outputs/train/$NAME"
mkdir -p "$OUT" "$REPO/logs"
cd "$REPO/upstream"
export DATA_ROOT="$REPO/data" WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
"$PY" train.py data_v2="$STREAM" \
  hydra.run.dir="$OUT" \
  lambda_flow=1.0 lambda_recon=0.5 lambda_velocity=0.5 lambda_acceleration=0.0 \
  "train_data=[How2Sign,CSL-Daily,PHOENIX-2014T]" "dev_data=[How2Sign]" "test_data=[How2Sign]" \
  learning_steps="$STEPS" save_interval=10000 eval_interval=5000 log_interval=500 \
  "$@" 2>&1 | tee "$REPO/logs/train_${NAME}.log"
