#!/usr/bin/env bash
set -euo pipefail

# Reproduce original UniTS point/reconstruction baselines for the UniTS-CQ plan.
#
# Examples:
#   bash scripts/reproduce_point_baseline.sh
#   TARGET=forecast DATASET=weather bash scripts/reproduce_point_baseline.sh
#   TARGET=forecast DATASET=ecl bash scripts/reproduce_point_baseline.sh
#   TARGET=ad DATASET=smd bash scripts/reproduce_point_baseline.sh
#   TARGET=ad DATASET=psm bash scripts/reproduce_point_baseline.sh
#
# Outputs:
#   Original UniTS logs/checkpoints:
#     outputs/point_baseline/
#   Exported .npz artifacts for conformal calibration:
#     outputs/units_cq/predictions/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${TARGET:-all}"      # all | forecast | ad
DATASET="${DATASET:-all}"    # all | weather | ecl | smd | psm
SEED="${SEED:-2021}"
WANDB_MODE="${WANDB_MODE:-disabled}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/point_baseline}"
ARTIFACT_DIR="${ARTIFACT_DIR:-outputs/units_cq/predictions}"
EXPORT_ARTIFACTS="${EXPORT_ARTIFACTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORECAST_CKPT="${FORECAST_CKPT:-checkpoints/units_x128_pretrain_checkpoint.pth}"
AD_CKPT="${AD_CKPT:-checkpoints/units_x32_pretrain_checkpoint.pth}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ANOMALY_RATIO="${ANOMALY_RATIO:-1.0}"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$ARTIFACT_DIR"

need_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "[missing] $path"
        return 1
    fi
    echo "[ok] $path"
}

check_forecast_data() {
    local missing=0
    if [ "$DATASET" = "all" ] || [ "$DATASET" = "weather" ]; then
        need_file "dataset/weather/weather.csv" || missing=1
    fi
    if [ "$DATASET" = "all" ] || [ "$DATASET" = "ecl" ]; then
        need_file "dataset/electricity/electricity.csv" || missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        echo
        echo "[error] Forecast data is missing. Run:"
        echo "        bash scripts/download_units_cq_data.sh"
        exit 1
    fi
}

check_ad_data() {
    local missing=0
    if [ "$DATASET" = "all" ] || [ "$DATASET" = "smd" ]; then
        need_file "dataset/SMD/SMD_train.npy" || missing=1
        need_file "dataset/SMD/SMD_test.npy" || missing=1
        need_file "dataset/SMD/SMD_test_label.npy" || missing=1
    fi
    if [ "$DATASET" = "all" ] || [ "$DATASET" = "psm" ]; then
        need_file "dataset/PSM/train.csv" || missing=1
        need_file "dataset/PSM/test.csv" || missing=1
        need_file "dataset/PSM/test_label.csv" || missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        echo
        echo "[error] AD data is missing. Run:"
        echo "        bash scripts/download_units_cq_data.sh"
        exit 1
    fi
}

run_torch() {
    local name="$1"
    shift
    local port="${MASTER_PORT:-$((RANDOM % 9000 + 1000))}"
    local log_file="$OUTPUT_DIR/logs/${name}.log"

    echo
    echo "[run] $name"
    echo "[log] $log_file"
    torchrun --nnodes 1 --nproc-per-node=1 --master_port "$port" run.py "$@" 2>&1 | tee "$log_file"
}

export_forecast_artifacts() {
    if [ "$EXPORT_ARTIFACTS" != "1" ]; then
        echo "[skip] EXPORT_ARTIFACTS=$EXPORT_ARTIFACTS"
        return
    fi

    if [ "$DATASET" = "all" ] || [ "$DATASET" = "weather" ]; then
        echo
        echo "[export] WeatherP96 point forecast artifact"
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task forecast \
            --dataset_key LTF_Weather_p96 \
            --task_data_config_path data_provider/baseline_forecast_weather_p96.yaml \
            --pretrained_weight "$FORECAST_CKPT" \
            --output_path "$ARTIFACT_DIR/weather_p96_units_point_seed${SEED}.npz" \
            --d_model 128 \
            --dropout 0.1 \
            --batch_size 1 \
            --seed "$SEED"
    fi

    if [ "$DATASET" = "all" ] || [ "$DATASET" = "ecl" ]; then
        echo
        echo "[export] ECLP96 point forecast artifact"
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task forecast \
            --dataset_key LTF_ECL_p96 \
            --task_data_config_path data_provider/baseline_forecast_ecl_p96.yaml \
            --pretrained_weight "$FORECAST_CKPT" \
            --output_path "$ARTIFACT_DIR/ecl_p96_units_point_seed${SEED}.npz" \
            --d_model 128 \
            --dropout 0.1 \
            --batch_size 1 \
            --seed "$SEED"
    fi
}

export_ad_artifacts() {
    if [ "$EXPORT_ARTIFACTS" != "1" ]; then
        echo "[skip] EXPORT_ARTIFACTS=$EXPORT_ARTIFACTS"
        return
    fi

    if [ "$DATASET" = "all" ] || [ "$DATASET" = "smd" ]; then
        echo
        echo "[export] SMD reconstruction-score artifact"
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task ad \
            --dataset_key SMD \
            --task_data_config_path data_provider/baseline_anomaly_smd.yaml \
            --pretrained_weight "$AD_CKPT" \
            --output_path "$ARTIFACT_DIR/smd_units_original_ad_seed${SEED}.npz" \
            --d_model 32 \
            --dropout 0.0 \
            --batch_size "$BATCH_SIZE" \
            --anomaly_ratio "$ANOMALY_RATIO" \
            --seed "$SEED"
    fi

    if [ "$DATASET" = "all" ] || [ "$DATASET" = "psm" ]; then
        echo
        echo "[export] PSM reconstruction-score artifact"
        "$PYTHON_BIN" scripts/export_point_artifacts.py \
            --task ad \
            --dataset_key PSM \
            --task_data_config_path data_provider/baseline_anomaly_psm.yaml \
            --pretrained_weight "$AD_CKPT" \
            --output_path "$ARTIFACT_DIR/psm_units_original_ad_seed${SEED}.npz" \
            --d_model 32 \
            --dropout 0.0 \
            --batch_size "$BATCH_SIZE" \
            --anomaly_ratio "$ANOMALY_RATIO" \
            --seed "$SEED"
    fi
}

run_forecast() {
    need_file "$FORECAST_CKPT"
    check_forecast_data

    local config="data_provider/baseline_forecast_point.yaml"
    local model_id="point_forecast_weather_ecl"
    if [ "$DATASET" = "weather" ]; then
        config="data_provider/baseline_forecast_weather_p96.yaml"
        model_id="point_forecast_weather_p96"
    elif [ "$DATASET" = "ecl" ]; then
        config="data_provider/baseline_forecast_ecl_p96.yaml"
        model_id="point_forecast_ecl_p96"
    fi

    run_torch "$model_id" \
        --is_training 0 \
        --fix_seed "$SEED" \
        --model_id "$model_id" \
        --model UniTS \
        --prompt_num 10 \
        --patch_len 16 \
        --stride 16 \
        --e_layers 3 \
        --d_model 128 \
        --des PointBaseline \
        --itr 1 \
        --batch_size 1 \
        --acc_it 1 \
        --dropout 0.1 \
        --debug "$WANDB_MODE" \
        --project_name units_cq_point_baseline \
        --pretrained_weight "$FORECAST_CKPT" \
        --checkpoints "$OUTPUT_DIR" \
        --task_data_config_path "$config"

    export_forecast_artifacts
}

run_ad() {
    need_file "$AD_CKPT"
    check_ad_data

    local config="data_provider/baseline_anomaly_point.yaml"
    local model_id="point_ad_smd_psm"
    if [ "$DATASET" = "smd" ]; then
        config="data_provider/baseline_anomaly_smd.yaml"
        model_id="point_ad_smd"
    elif [ "$DATASET" = "psm" ]; then
        config="data_provider/baseline_anomaly_psm.yaml"
        model_id="point_ad_psm"
    fi

    run_torch "$model_id" \
        --is_training 0 \
        --fix_seed "$SEED" \
        --model_id "$model_id" \
        --model UniTS \
        --prompt_num 10 \
        --patch_len 16 \
        --stride 16 \
        --e_layers 3 \
        --d_model 32 \
        --des PointBaseline \
        --itr 1 \
        --batch_size "$BATCH_SIZE" \
        --acc_it 1 \
        --dropout 0.0 \
        --debug "$WANDB_MODE" \
        --project_name units_cq_point_baseline \
        --pretrained_weight "$AD_CKPT" \
        --checkpoints "$OUTPUT_DIR" \
        --anomaly_ratio "$ANOMALY_RATIO" \
        --task_data_config_path "$config"

    export_ad_artifacts
}

case "$TARGET" in
    all)
        if [ "$DATASET" != "all" ]; then
            echo "[error] TARGET=all only supports DATASET=all."
            echo "        Use TARGET=forecast or TARGET=ad for a single dataset."
            exit 1
        fi
        run_forecast
        run_ad
        ;;
    forecast)
        case "$DATASET" in
            all|weather|ecl) run_forecast ;;
            *) echo "[error] For TARGET=forecast, DATASET must be all, weather, or ecl."; exit 1 ;;
        esac
        ;;
    ad)
        case "$DATASET" in
            all|smd|psm) run_ad ;;
            *) echo "[error] For TARGET=ad, DATASET must be all, smd, or psm."; exit 1 ;;
        esac
        ;;
    *)
        echo "[error] TARGET must be all, forecast, or ad."
        exit 1
        ;;
esac

echo
echo "[done] Point baseline run finished. Logs are under $OUTPUT_DIR/logs"
if [ "$EXPORT_ARTIFACTS" = "1" ]; then
    echo "[done] Exported .npz artifacts are under $ARTIFACT_DIR"
fi
