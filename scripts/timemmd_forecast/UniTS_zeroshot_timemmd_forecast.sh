#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CKPT=checkpoints/<zeroshot-checkpoint>.pth \
#     bash scripts/timemmd_forecast/UniTS_zeroshot_timemmd_forecast.sh

ckpt_path=${CKPT:-}
model_name=${MODEL_NAME:-UniTS_zeroshot}
exp_name=${EXP_NAME:-zeroshot_timemmd_forecast}
project_name=${PROJECT_NAME:-timemmd_forecast}
result_root=${RESULT_ROOT:-results}
result_tag=${RESULT_TAG:-timemmd_forecast}
run_name=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
seed=${SEED:-2021}
patch_len=${PATCH_LEN:-6}
stride=${STRIDE:-6}
prompt_num=${PROMPT_NUM:-10}
d_model=${D_MODEL:-32}
e_layers=${E_LAYERS:-3}
random_port=${MASTER_PORT:-$((RANDOM % 9000 + 1000))}
run_dir="${result_root}/${result_tag}/${exp_name}/${run_name}/seed_${seed}"

if [[ -z "$ckpt_path" || ! -f "$ckpt_path" ]]; then
  echo "Set CKPT to an existing UniTS_zeroshot checkpoint." >&2
  echo "Example: CKPT=checkpoints/units_zeroshot_x32.pth bash $0" >&2
  exit 2
fi

mkdir -p "$run_dir"

torchrun --nnodes=1 --nproc-per-node=1 --master_port="$random_port" run.py \
  --is_training 0 \
  --fix_seed "$seed" \
  --model_id "$exp_name" \
  --model "$model_name" \
  --pretrained_weight "$ckpt_path" \
  --features S \
  --target OT \
  --prompt_num "$prompt_num" \
  --patch_len "$patch_len" \
  --stride "$stride" \
  --e_layers "$e_layers" \
  --d_model "$d_model" \
  --batch_size "${BATCH_SIZE:-32}" \
  --dropout "${DROPOUT:-0}" \
  --des Exp \
  --debug "${WANDB_MODE:-disabled}" \
  --project_name "$project_name" \
  --no_memory_check \
  --compact_log \
  --result_dir "$run_dir" \
  --task_data_config_path data_provider/timemmd_forecast_prompt.yaml \
  2>&1 | tee "$run_dir/run.log"
