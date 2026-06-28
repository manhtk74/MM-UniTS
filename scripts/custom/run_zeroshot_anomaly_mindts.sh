#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_NAME="${MODEL_NAME:-UniTS}"
EXP_NAME="${EXP_NAME:-mindts_anomaly_zeroshot}"
PROJECT_NAME="${PROJECT_NAME:-mindts_anomaly_zeroshot}"
WANDB_MODE="${WANDB_MODE:-disabled}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CONFIG_PATH="${CONFIG_PATH:-data_provider/custom_mindts_anomaly.yaml}"
CKPT_PATH="${CKPT_PATH:-./checkpoints/units_x32_pretrain_checkpoint.pth}"
D_MODEL="${D_MODEL:-}"
if [[ -z "$D_MODEL" ]]; then
  if [[ "$CKPT_PATH" == *"x128"* ]]; then
    D_MODEL=128
  elif [[ "$CKPT_PATH" == *"x64"* ]]; then
    D_MODEL=64
  elif [[ "$CKPT_PATH" == *"x32"* ]]; then
    D_MODEL=32
  else
    D_MODEL=32
  fi
fi
ANOMALY_RATIO="${ANOMALY_RATIO:-11.25}"
PATCH_LEN="${PATCH_LEN:-6}"
STRIDE="${STRIDE:-6}"
PORT="${PORT:-$((RANDOM % 9000 + 1000))}"

torchrun --nnodes 1 --nproc-per-node 1 --master_port "$PORT" run.py \
  --is_training 0 \
  --fix_seed 2021 \
  --model_id "$EXP_NAME" \
  --pretrained_weight "$CKPT_PATH" \
  --model "$MODEL_NAME" \
  --prompt_num 10 \
  --patch_len "$PATCH_LEN" \
  --stride "$STRIDE" \
  --e_layers 3 \
  --d_model "$D_MODEL" \
  --des ZeroShot \
  --batch_size "$BATCH_SIZE" \
  --dropout 0.0 \
  --debug "$WANDB_MODE" \
  --project_name "$PROJECT_NAME" \
  --anomaly_ratio "$ANOMALY_RATIO" \
  --freq d \
  --num_workers 0 \
  --task_data_config_path "$CONFIG_PATH"
