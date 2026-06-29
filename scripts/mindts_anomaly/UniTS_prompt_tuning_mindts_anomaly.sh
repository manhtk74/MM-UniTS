#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-UniTS}
wandb_mode=${WANDB_MODE:-disabled}
project_name=${PROJECT_NAME:-mindts_anomaly_detection}
exp_name=${EXP_NAME:-prompt_tuning_mindts_anomaly_pct05}
ckpt_path=${CKPT:-checkpoints/units_x32_pretrain_checkpoint.pth}
result_root=${RESULT_ROOT:-results/mindts_prompt}
seed=${SEED:-2021}
run_name=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
run_dir="${result_root}/${exp_name}/${run_name}/seed_${seed}"
random_port=$((RANDOM % 9000 + 1000))

mkdir -p "$run_dir"

torchrun --nnodes 1 --nproc-per-node=1 --master_port "$random_port" run.py \
  --fix_seed "$seed" \
  --is_training 1 \
  --subsample_pct "${SUBSAMPLE_PCT:-0.05}" \
  --model_id "$exp_name" \
  --pretrained_weight "$ckpt_path" \
  --model "$model_name" \
  --prompt_num "${PROMPT_NUM:-10}" \
  --patch_len "${PATCH_LEN:-6}" \
  --stride "${STRIDE:-6}" \
  --e_layers "${E_LAYERS:-3}" \
  --d_model "${D_MODEL:-32}" \
  --des 'Exp' \
  --itr 1 \
  --lradj prompt_tuning \
  --learning_rate "${LR:-5e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-3}" \
  --train_epochs 0 \
  --prompt_tune_epoch "${PROMPT_EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --acc_it "${ACC_IT:-4}" \
  --dropout "${DROPOUT:-0}" \
  --debug "$wandb_mode" \
  --project_name "$project_name" \
  --clip_grad "${CLIP_GRAD:-100}" \
  --no_memory_check \
  --compact_log \
  --result_dir "$run_dir" \
  --anomaly_score_mode "${SCORE_MODE:-thre}" \
  --anomaly_ratios ${ANOMALY_RATIOS:-0.1 0.5 1 2 3 4 5 6 7 8 9 10 12 15 20 25 30 35 40 45 50} \
  --task_data_config_path data_provider/mindts_anomaly_detection.yaml \
  2>&1 | tee "$run_dir/run.log"
