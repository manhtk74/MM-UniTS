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
with the MindTS metric implementation: `Aff-F`, `V-PR`, and `V-ROC`.

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

The script writes:

```bash
results/mindts_units_anomaly.csv
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

| Dataset | Aff-F | V-PR | V-ROC |
| --- | ---: | ---: | ---: |
| Weather | 76.17 | 44.35 | 75.08 |
| Energy | 63.84 | 31.04 | 51.15 |
| Environment | 83.06 | 50.24 | 92.03 |
| KR | 82.24 | 43.32 | 73.93 |
| EWJ | 77.61 | 39.32 | 73.91 |
| MDT | 75.57 | 37.61 | 58.67 |

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
