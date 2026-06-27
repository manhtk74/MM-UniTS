#!/usr/bin/env bash
set -euo pipefail

# Export original UniTS point/reconstruction artifacts to .npz.
#
# Examples:
#   TASK=forecast DATASET=weather bash scripts/export_point_artifacts.sh
#   TASK=forecast DATASET=ecl bash scripts/export_point_artifacts.sh
#   TASK=ad DATASET=smd bash scripts/export_point_artifacts.sh
#   TASK=ad DATASET=psm bash scripts/export_point_artifacts.sh
#
# For a tiny smoke run:
#   TASK=ad DATASET=smd MAX_BATCHES=2 bash scripts/export_point_artifacts.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-ad}"          # forecast | ad
DATASET="${DATASET:-smd}"   # weather | ecl | smd | psm
SEED="${SEED:-2021}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/units_cq/predictions}"
MAX_BATCHES="${MAX_BATCHES:-}"

mkdir -p "$OUTPUT_DIR"

extra_args=()
if [ -n "$MAX_BATCHES" ]; then
    extra_args+=(--max_batches "$MAX_BATCHES")
fi

case "${TASK}:${DATASET}" in
    forecast:weather)
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task forecast \
            --dataset_key LTF_Weather_p96 \
            --task_data_config_path data_provider/baseline_forecast_weather_p96.yaml \
            --pretrained_weight checkpoints/units_x128_pretrain_checkpoint.pth \
            --output_path "$OUTPUT_DIR/weather_p96_units_point_seed${SEED}.npz" \
            --d_model 128 \
            --dropout 0.1 \
            --batch_size 1 \
            --seed "$SEED" \
            "${extra_args[@]}"
        ;;
    forecast:ecl)
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task forecast \
            --dataset_key LTF_ECL_p96 \
            --task_data_config_path data_provider/baseline_forecast_ecl_p96.yaml \
            --pretrained_weight checkpoints/units_x128_pretrain_checkpoint.pth \
            --output_path "$OUTPUT_DIR/ecl_p96_units_point_seed${SEED}.npz" \
            --d_model 128 \
            --dropout 0.1 \
            --batch_size 1 \
            --seed "$SEED" \
            "${extra_args[@]}"
        ;;
    ad:smd)
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task ad \
            --dataset_key SMD \
            --task_data_config_path data_provider/baseline_anomaly_smd.yaml \
            --pretrained_weight checkpoints/units_x32_pretrain_checkpoint.pth \
            --output_path "$OUTPUT_DIR/smd_units_original_ad_seed${SEED}.npz" \
            --d_model 32 \
            --dropout 0.0 \
            --batch_size 32 \
            --seed "$SEED" \
            "${extra_args[@]}"
        ;;
    ad:psm)
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task ad \
            --dataset_key PSM \
            --task_data_config_path data_provider/baseline_anomaly_psm.yaml \
            --pretrained_weight checkpoints/units_x32_pretrain_checkpoint.pth \
            --output_path "$OUTPUT_DIR/psm_units_original_ad_seed${SEED}.npz" \
            --d_model 32 \
            --dropout 0.0 \
            --batch_size 32 \
            --seed "$SEED" \
            "${extra_args[@]}"
        ;;
    *)
        echo "[error] Unsupported TASK/DATASET: ${TASK}/${DATASET}"
        echo "        Use forecast/weather, forecast/ecl, ad/smd, or ad/psm."
        exit 1
        ;;
esac

