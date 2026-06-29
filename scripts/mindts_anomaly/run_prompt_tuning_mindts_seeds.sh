#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-2021 2022 2023 2024 2025}"
EXP_NAME="${EXP_NAME:-prompt_tuning_mindts_anomaly_pct05}"
RESULT_ROOT="${RESULT_ROOT:-results/mindts_prompt}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"

echo "Output folder: $RESULT_ROOT/$EXP_NAME/$RUN_NAME"

for seed in $SEEDS; do
  echo "===== MindTS prompt tuning seed $seed ====="
  SEED="$seed" EXP_NAME="$EXP_NAME" RESULT_ROOT="$RESULT_ROOT" RUN_NAME="$RUN_NAME" \
    bash scripts/mindts_anomaly/UniTS_prompt_tuning_mindts_anomaly.sh
done

python scripts/mindts_anomaly/summarize_mindts_results.py \
  "$RESULT_ROOT/$EXP_NAME/$RUN_NAME"/seed_*/mindts_seed*.csv \
  --output "$RESULT_ROOT/$EXP_NAME/$RUN_NAME/summary_best_rows.csv" \
  --summary-output "$RESULT_ROOT/$EXP_NAME/$RUN_NAME/summary_mean_std.csv"
