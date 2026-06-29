#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-2021 2022 2023 2024 2025}"
EXP_NAME="${EXP_NAME:-prompt_tuning_timemmd_forecast}"
RESULT_ROOT="${RESULT_ROOT:-results}"
RESULT_TAG="${RESULT_TAG:-timemmd_forecast}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$RESULT_ROOT/$RESULT_TAG/$EXP_NAME/$RUN_NAME"

echo "Output folder: $RUN_ROOT"

for seed in $SEEDS; do
  echo "===== Time-MMD prompt tuning seed $seed ====="
  SEED="$seed" EXP_NAME="$EXP_NAME" RESULT_ROOT="$RESULT_ROOT" RESULT_TAG="$RESULT_TAG" RUN_NAME="$RUN_NAME" \
    bash scripts/timemmd_forecast/UniTS_prompt_tuning_timemmd_forecast.sh
done

python scripts/timemmd_forecast/summarize_timemmd_forecast.py \
  "$RUN_ROOT"/seed_*/results_seed*.csv \
  --output "$RUN_ROOT/summary_best_rows.csv" \
  --summary-output "$RUN_ROOT/summary_mean_std.csv"
