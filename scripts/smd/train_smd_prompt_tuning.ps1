param(
    [string]$PretrainedWeight = "checkpoints/units_x32_pretrain_checkpoint.pth",
    [string]$ModelId = "smd_demo_prompt_tuning",
    [int]$PromptEpochs = 10
)

# Run from the UniTS directory in a CUDA-enabled environment.
python -m torch.distributed.run --nproc_per_node=1 run.py `
  --fix_seed 2021 `
  --is_training 1 `
  --subsample_pct 0.05 `
  --model_id $ModelId `
  --pretrained_weight $PretrainedWeight `
  --model UniTS `
  --prompt_num 10 `
  --patch_len 16 `
  --stride 16 `
  --e_layers 3 `
  --d_model 32 `
  --n_heads 8 `
  --des SMD_Demo `
  --itr 1 `
  --lradj prompt_tuning `
  --learning_rate 5e-5 `
  --weight_decay 1e-2 `
  --train_epochs 0 `
  --prompt_tune_epoch $PromptEpochs `
  --batch_size 32 `
  --acc_it 32 `
  --dropout 0.0 `
  --debug disabled `
  --project_name smd_demo `
  --clip_grad 100 `
  --no_memory_check `
  --result_dir results/smd_demo `
  --task_data_config_path data_provider/smd_demo.yaml
