#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-UniTS}
wandb_mode=${WANDB_MODE:-disabled}
project_name=${PROJECT_NAME:-timemmd_forecast}
exp_name=${EXP_NAME:-prompt_tuning_timemmd_forecast}
ckpt_path=${CKPT:-checkpoints/units_x32_pretrain_checkpoint.pth}
result_root=${RESULT_ROOT:-results}
result_tag=${RESULT_TAG:-timemmd_forecast}
seed=${SEED:-2021}
run_name=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
run_dir="${result_root}/${result_tag}/${exp_name}/${run_name}/seed_${seed}"
random_port=$((RANDOM % 9000 + 1000))
text_args=()
if [[ "${USE_TEXT_MODALITY:-0}" == "1" ]]; then
  text_args+=(--use_text_modality --text_embedding_dim "${TEXT_EMBEDDING_DIM:-768}")
fi

mkdir -p "$run_dir"

torchrun --nnodes 1 --nproc-per-node=1 --master_port "$random_port" run.py \
  --fix_seed "$seed" \
  --is_training 1 \
  --subsample_pct "${SUBSAMPLE_PCT:-1.0}" \
  --model_id "$exp_name" \
  --pretrained_weight "$ckpt_path" \
  --model "$model_name" \
  --features S \
  --target OT \
  --prompt_num "${PROMPT_NUM:-10}" \
  --patch_len "${PATCH_LEN:-4}" \
  --stride "${STRIDE:-4}" \
  --e_layers "${E_LAYERS:-3}" \
  --d_model "${D_MODEL:-32}" \
  "${text_args[@]}" \
  --des 'Exp' \
  --itr 1 \
  --lradj prompt_tuning \
  --learning_rate "${LR:-5e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-3}" \
  --train_epochs 0 \
  --prompt_tune_epoch "${PROMPT_EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --acc_it "${ACC_IT:-8}" \
  --dropout "${DROPOUT:-0}" \
  --debug "$wandb_mode" \
  --project_name "$project_name" \
  --clip_grad "${CLIP_GRAD:-100}" \
  --no_memory_check \
  --compact_log \
  --result_dir "$run_dir" \
  --task_data_config_path data_provider/timemmd_forecast_prompt.yaml \
  2>&1 | tee "$run_dir/run.log"
