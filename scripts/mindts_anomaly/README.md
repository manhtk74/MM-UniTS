# UniTS on MindTS anomaly detection

This runner reproduces the numerical-only UniTS baseline from the MindTS paper
on the six MindTS anomaly datasets:

- Weather
- Energy
- Environment
- KR
- EWJ
- MDT

It reads MindTS OTB-format CSV files directly from `../MindTS/dataset/anomaly_detect`,
trains UniTS with the few-shot anomaly settings used by the UniTS scripts, and evaluates
with the MindTS metric implementation. The output includes point-wise `Precision`,
`Recall`, `F1`, plus `Aff-F`, `V-PR`, and `V-ROC`.

## Kaggle layout

Place or clone both repositories under the same parent directory:

```bash
/kaggle/working/UniTS
/kaggle/working/MindTS
```

Run from the UniTS root:

```bash
bash scripts/mindts_anomaly/run_units_mindts_kaggle.sh
```

To launch the full tuning grid:

```bash
bash scripts/mindts_anomaly/tune_units_mindts_all.sh
```

The script writes:

```bash
results/mindts_units_anomaly.csv
```

Each row is one dataset and anomaly-ratio threshold, with columns:

```bash
dataset, ratio, threshold, Precision, Recall, F1, Aff-F, V-PR, V-ROC
```

## Checkpoint

For the paper-like setting, put the UniTS x32 pretrained checkpoint here:

```bash
newcheckpoints/units_x32_pretrain_checkpoint.pth
```

The shell script automatically uses it if present. If it is missing, the run falls
back to random initialization and should not be compared directly with the paper.

## Paper Table 1 targets

The runner prints the MindTS paper's UniTS Table 1 values next to each reproduced
dataset result:

| Dataset | Precision | Recall | F1 | Aff-F | V-PR | V-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weather | 35.88 | 82.46 | 50.00 | 76.17 | 44.35 | 75.08 |
| Energy | 20.20 | 73.21 | 31.66 | 63.84 | 31.04 | 51.15 |
| Environment | 35.96 | 83.33 | 50.24 | 83.06 | 50.24 | 92.03 |
| KR | 30.23 | 79.79 | 43.84 | 82.24 | 43.32 | 73.93 |
| EWJ | 26.95 | 71.70 | 39.18 | 77.61 | 39.32 | 73.91 |
| MDT | 44.19 | 62.30 | 51.70 | 75.57 | 37.61 | 58.67 |

## Useful overrides

```bash
MINDTS_ROOT=/kaggle/working/MindTS \
CKPT=/kaggle/working/UniTS/newcheckpoints/units_x32_pretrain_checkpoint.pth \
OUT=results/my_units_mindts.csv \
bash scripts/mindts_anomaly/run_units_mindts_kaggle.sh
```

For full-data finetuning instead of UniTS few-shot:

```bash
python scripts/mindts_anomaly/reproduce_units_mindts.py \
  --mindts-root ../MindTS \
  --checkpoint newcheckpoints/units_x32_pretrain_checkpoint.pth \
  --subsample-pct 1.0 \
  --output results/mindts_units_full_train.csv
```
