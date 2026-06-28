#!/usr/bin/env bash
set -euo pipefail

# Run from the UniTS/MM-UniTS repository root:
#   bash scripts/mindts_anomaly/tune_units_mindts_all.sh
#
# Optional overrides:
#   MINDTS_ROOT=dataset/MindTS
#   CKPT=newcheckpoints/units_x32_pretrain_checkpoint.pth
#   DATASETS="Weather Energy Environment KR EWJ MDT"
#   OUT_DIR=results/tune_units_mindts
#   FORCE=1

MINDTS_ROOT="${MINDTS_ROOT:-dataset/MindTS}"
CKPT="${CKPT:-checkpoints/units_x32_pretrain_checkpoint.pth}"
DATASETS="${DATASETS:-Weather Energy Environment KR EWJ MDT}"
OUT_DIR="${OUT_DIR:-results/tune_units_mindts}"
FORCE="${FORCE:-0}"

RATIOS="${RATIOS:-0.1 0.5 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50}"
BASE_ARGS=(
  --mindts-root "$MINDTS_ROOT"
  --datasets $DATASETS
  --epochs 10
  --subsample-pct 0.05
  --seq-len 96
  --patch-len 16
  --stride 16
  --learning-rate 5e-4
  --weight-decay 1e-3
  --anomaly-ratios $RATIOS
)

if [[ -f "$CKPT" ]]; then
  BASE_ARGS+=(--checkpoint "$CKPT")
else
  echo "Warning: checkpoint not found at $CKPT; runs will use random initialization."
fi

mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  shift
  local output="$OUT_DIR/${name}.csv"
  local log="$OUT_DIR/${name}.log"

  if [[ "$FORCE" != "1" && -f "$output" ]]; then
    echo "Skip existing: $output"
    return 0
  fi

  echo
  echo "===== RUN $name ====="
  echo "Output: $output"
  python scripts/mindts_anomaly/reproduce_units_mindts.py "$@" --output "$output" 2>&1 | tee "$log"
}

echo "MindTS root: $MINDTS_ROOT"
echo "Checkpoint:  $CKPT"
echo "Datasets:    $DATASETS"
echo "Out dir:     $OUT_DIR"

# 1) Fair checkpoint-compatible seq_len tuning.
# Keep patch_len=stride=16 so patch embedding/head can load from UniTS checkpoint.
for S in 32 48 64 96; do
  run_one "fair_seq${S}" \
    "${BASE_ARGS[@]}" \
    --seq-len "$S" \
    --patch-len 16 \
    --stride 16
done

# 2) Few-shot percentage tuning.
for PCT in 0.05 0.1 0.2 1.0; do
  pct_name="${PCT/./p}"
  run_one "pct${pct_name}_seq96_patch16" \
    "${BASE_ARGS[@]}" \
    --subsample-pct "$PCT" \
    --seq-len 96 \
    --patch-len 16 \
    --stride 16
done

# 3) Learning-rate and epoch tuning.
for LR in 1e-4 3e-4 5e-4; do
  for E in 5 10 15; do
    lr_name="${LR/-/m}"
    run_one "lr${lr_name}_ep${E}_seq96_patch16" \
      "${BASE_ARGS[@]}" \
      --epochs "$E" \
      --learning-rate "$LR" \
      --seq-len 96 \
      --patch-len 16 \
      --stride 16
  done
done

# 4) Aggressive tuned windows.
# These may skip patch embedding/head tensors from the checkpoint if patch_len != 16.
for SPEC in \
  "24 6 6" \
  "48 12 12" \
  "96 16 16"; do
  read -r S P ST <<< "$SPEC"
  run_one "aggressive_seq${S}_patch${P}_stride${ST}" \
    "${BASE_ARGS[@]}" \
    --seq-len "$S" \
    --patch-len "$P" \
    --stride "$ST"
done

echo
echo "All tuning runs finished. CSV/log files are in: $OUT_DIR"
