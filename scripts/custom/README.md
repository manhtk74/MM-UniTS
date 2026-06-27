# Custom UniTS Baselines

This folder contains the local baseline pipeline for:

- Time-MMD numerical forecasting with `OT` as the default target.
- MindTS anomaly detection datasets converted to generic `.npy` windows.

Run commands from `UniTS/` or call the shell scripts directly.

## 1. Prepare data

```bash
bash scripts/custom/prepare_data.sh
```

Outputs:

- `dataset/TimeMMD/<Domain>/<Domain>.csv`
- `dataset/MindTS_AD/<Domain>/train.npy`
- `dataset/MindTS_AD/<Domain>/test.npy`
- `dataset/MindTS_AD/<Domain>/test_label.npy`
- `dataset/custom_data_manifest.json`
- regenerated YAML configs in `data_provider/`

By default this processes every available dataset:

- Time-MMD numerical: Agriculture, Climate, Economy, Energy, Environment, Health_AFR, Health_US, Security, SocialGood, Traffic
- MindTS anomaly detection: EWJ, Energy, Environment, KR, MDT, Weather

## 2. Smoke check loaders

```bash
python scripts/custom/smoke_check_custom_data.py
```

This verifies that the YAML configs can instantiate UniTS dataloaders and prints batch shapes.

## 3. Put pretrained checkpoint in place

Prompt tuning uses an existing UniTS pretrained checkpoint. The default scripts expect:

```text
newcheckpoints/units_x64_pretrain_checkpoint.pth
```

You can also pass any checkpoint path via `CKPT_PATH=...`.

## 4. Prompt tune all tasks together

```bash
CKPT_PATH=./newcheckpoints/units_x64_pretrain_checkpoint.pth \
bash scripts/custom/run_prompt_all_custom.sh
```

Default config: `data_provider/custom_all_prompt.yaml`.

## 5. Prompt tune forecasting only

```bash
CKPT_PATH=./newcheckpoints/units_x64_pretrain_checkpoint.pth \
bash scripts/custom/run_prompt_forecast_timemmd.sh
```

Default config: `data_provider/custom_timemmd_forecast.yaml`.

## 6. Prompt tune anomaly detection only

```bash
ANOMALY_RATIO=11.25 \
CKPT_PATH=./newcheckpoints/units_x64_pretrain_checkpoint.pth \
bash scripts/custom/run_prompt_anomaly_mindts.sh
```

Default config: `data_provider/custom_mindts_anomaly.yaml`.

The six prepared MindTS anomaly datasets average about 11.25 percent anomaly labels, so `ANOMALY_RATIO=11.25` is a better first global setting than UniTS' default `1.0`. For single-dataset runs, use the ratio in `dataset/MindTS_AD/<Domain>/metadata.json`.

## Optional local pretrain

`run_pretrain_timemmd.sh` is kept only as an optional local pretrain script. The main baseline path above uses a pretrained checkpoint directly.
