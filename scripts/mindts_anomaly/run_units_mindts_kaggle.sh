#!/usr/bin/env bash
set -euo pipefail

# Run from the UniTS repository root:
#   bash scripts/mindts_anomaly/run_units_mindts_kaggle.sh
#
# Expected dataset layout:
#   UniTS/dataset/MindTS/DETECT_META.csv
#   UniTS/dataset/MindTS/Weather.csv
#
# Optional checkpoint:
#   UniTS/newcheckpoints/units_x32_pretrain_checkpoint.pth

MINDTS_ROOT="${MINDTS_ROOT:-dataset/MindTS}"
CKPT="${CKPT:-checkpoints/units_x32_pretrain_checkpoint.pth}"
OUT="${OUT:-results/mindts_units_anomaly.csv}"

CKPT_ARG=()
if [[ -f "$CKPT" ]]; then
  CKPT_ARG=(--checkpoint "$CKPT")
else
  echo "Warning: checkpoint not found at $CKPT; running from random initialization."
fi

python scripts/mindts_anomaly/reproduce_units_mindts.py \
  --mindts-root "$MINDTS_ROOT" \
  "${CKPT_ARG[@]}" \
  --datasets Weather Energy Environment KR EWJ MDT \
  --epochs 10 \
  --subsample-pct 0.05 \
  --batch-size 32 \
  --acc-it 32 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3 \
  --seq-len 96 \
  --patch-len 16 \
  --stride 16 \
  --prompt-num 10 \
  --d-model 32 \
  --e-layers 3 \
  --dropout 0.0 \
  --score-mode thre \
  --output "$OUT"
