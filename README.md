# MM-UniTS: Multimodal Unified Time Series Model

[**Original Project Page**](https://zitniklab.hms.harvard.edu/projects/UniTS/)  |   [**UniTS Paper**](https://arxiv.org/pdf/2403.00131.pdf) **(NeurIPS 2024)**

MM-UniTS extends the [UniTS](https://github.com/mims-harvard/UniTS) foundation model with **multimodal capabilities**, enabling it to jointly leverage numerical time series and textual side-information (event logs, alarm descriptions, ticket text) through a lightweight cross-attention adapter. The model supports forecasting, classification, imputation, and anomaly detection — all with shared parameters.

Authors: [Shanghua Gao](https://shgao.site/) [Teddy Koker](https://teddykoker.com) [Owen Queen](https://owencqueen.github.io/) [Thomas Hartvigsen](https://www.tomhartvigsen.com/) [Theodoros Tsiligkaridis](https://sites.google.com/view/theo-t) [Marinka Zitnik](https://zitniklab.hms.harvard.edu/)

<p align="center">
    <img src="https://zitniklab.hms.harvard.edu/img/UniTS-1.png" alt="UniTS-1" width="500">
</p>

---

## Table of Contents

- [Overview](#overview)
- [What's New: Multimodal Extension](#whats-new-multimodal-extension)
- [Setup](#setup)
- [Datasets](#datasets)
- [Quickstart](#quickstart)
- [Multimodal Anomaly Detection (MindTS)](#multimodal-anomaly-detection-mindts)
- [Forecasting (Time-MMD)](#forecasting-time-mmd)
- [Original UniTS Tasks](#original-units-tasks)
- [CLI Reference](#cli-reference)
- [Architecture Details](#architecture-details)
- [Pretrained Weights](#pretrained-weights)
- [Citation](#citation)

---

## Overview

Foundation models are profoundly transforming deep learning. Instead of training many task-specific models, a single pretrained model can be adapted to many tasks via few-shot prompting or fine-tuning. However, current foundation models apply to sequence data but not to time series, which present unique challenges due to diverse multi-domain datasets and diverging task specifications.

UniTS is a unified time series model that supports a universal task specification, accommodating classification, forecasting, imputation, and anomaly detection tasks. It achieves this through a novel unified network backbone incorporating sequence and variable attention along with a dynamic linear operator.

Across 38 multi-domain datasets, UniTS demonstrates superior performance compared to task-specific models and repurposed natural language-based LLMs, exhibiting remarkable zero-shot, few-shot, and prompt learning capabilities.

---

## Multimodal Extension

This fork extends UniTS with **multimodal fusion** for datasets that pair numerical time series with textual metadata (e.g., event logs, alarm descriptions, incident tickets). The key additions are:

### Multimodal Datasets

| Dataset | Type | Domains | Modalities |
|---------|------|---------|------------|
| [**Time-MMD**](https://github.com/AdityaLab/Time-MMD) | Forecasting | Agriculture, Climate, Economy, Energy, Environment, Health, Security, SocialGood, Traffic | Numerical + Text |
| [**MindTS**](https://github.com/AdityaLab/MindTS) | Anomaly Detection | EWJ, Energy, Environment, KR, MDT, Weather | Numerical + Text |

### Text Adapter Architecture

A lightweight **cross-attention adapter** fuses text information into UniTS without modifying the frozen backbone:

```
x_window
  -> UniTS patch embedding
  -> frozen UniTS backbone
  -> H  (hidden sample tokens)

text_by_timestep
  -> frozen text encoder / precomputed embeddings
  -> patch-aligned pooling
  -> text projector
  -> T  (text tokens)

H_fused = H + sigmoid(gate) * CrossAttn(LN(H), LN(T), LN(T))

x_hat = frozen GEN tower(H_fused)
score = reconstruction_error(x_hat, x_window)
```

### Two-Stage Training

1. **Text Adapter Warmup** — Train only the text projector, cross-attention, and gating parameter on top of a frozen pretrained UniTS checkpoint.
2. **Prompt Tuning** — Jointly train soft prompt tokens and the text adapter while keeping the backbone and generation head frozen.

---

## Setup

### 1. Requirements

Install PyTorch 2.0+ and the required packages:

```bash
pip install -r requirements.txt
```

Key additional dependencies for the multimodal pipeline:
- `sentence-transformers` (for precomputing text embeddings)
- `wandb` (optional, for experiment tracking)

### 2. Prepare Multimodal Data

Run the unified data preparation script:

```bash
bash scripts/custom/prepare_data.sh
```

This script:
- Downloads and converts **Time-MMD** numerical forecasting datasets (10 domains)
- Downloads and converts **MindTS** anomaly detection datasets (6 domains) with aligned text
- Generates YAML configs under `data_provider/`
- Produces a `dataset/custom_data_manifest.json` inventory

#### Output Structure

```
dataset/
├── TimeMMD/
│   ├── Agriculture/Agriculture.csv
│   ├── Climate/Climate.csv
│   └── ...
└── MindTS_AD/
    ├── EWJ/
    │   ├── train.npy              # numeric train windows
    │   ├── test.npy               # numeric test windows
    │   ├── test_label.npy         # anomaly labels
    │   ├── train_text.jsonl       # aligned text per timestep
    │   ├── test_text.jsonl
    │   ├── train_text_emb.npy     # precomputed text embeddings [T, E]
    │   ├── test_text_emb.npy
    │   └── metadata.json          # domain-specific anomaly ratio
    ├── Energy/
    └── ...
```

### 3. Precompute Text Embeddings

Text embeddings are precomputed:

```bash
python scripts/custom/precompute_mindts_text_embeddings.py
```

### 4. Smoke-Check Data Loaders

Verify that all YAML configs and data loaders work:

```bash
python scripts/custom/smoke_check_custom_data.py
```

### 5. Obtain a Pretrained Checkpoint

The multimodal pipeline builds on a pretrained UniTS checkpoint. Place it at:

```
checkpoints/units_x64_pretrain_checkpoint.pth
```

Pretrained weights are available at [UniTS Releases](https://github.com/mims-harvard/UniTS/releases/tag/ckpt). You can also pass any checkpoint path via `CKPT_PATH=...`.

---

## Quickstart

### All Tasks Together (Forecasting + Anomaly Detection)

```bash
CKPT_PATH=./checkpoints/units_x64_pretrain_checkpoint.pth \
bash scripts/custom/run_prompt_all_custom.sh
```

Default config: `data_provider/custom_all_prompt.yaml`

---

## Multimodal Anomaly Detection (MindTS)

The MindTS anomaly detection pipeline uses a **two-stage training** approach:

### Stage 1: Text Adapter Warmup

Train only the text bridge (projector + cross-attention + gate) from a pretrained UniTS checkpoint:

```bash
ANOMALY_RATIO=11.25 \
CKPT_PATH=./checkpoints/units_x64_pretrain_checkpoint.pth \
TEXT_WARMUP_EPOCHS=5 \
bash scripts/custom/run_train_text_adapter_anomaly_mindts.sh
```

### Stage 2: Prompt Tuning with Text Adapter

From the adapter-warmup checkpoint, jointly train prompt tokens and text adapter:

```bash
ANOMALY_RATIO=11.25 \
ADAPTER_CKPT_PATH=./checkpoints/ALL_task_mindts_anomaly_text_adapter_UniTS_All_ftM_dm64_el3_Exp_0/text_warmup_checkpoint.pth \
bash scripts/custom/run_prompt_anomaly_mindts.sh
```

### Zero-Shot Anomaly Detection

Evaluate a pretrained checkpoint directly without any training:

```bash
ANOMALY_RATIO=11.25 \
CKPT_PATH=./checkpoints/units_x32_pretrain_checkpoint.pth \
bash scripts/custom/run_zeroshot_anomaly_mindts.sh
```

### Configuration Notes

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ANOMALY_RATIO` | 11.25 | Average anomaly ratio across the 6 MindTS domains. For single-domain runs, use the ratio in `dataset/MindTS_AD/<Domain>/metadata.json` |
| `seq_len` | 24 | Input window length (set in YAML) |
| `patch_len` | 16 | Must match the pretrained checkpoint |
| `stride` | 16 | Must match the pretrained checkpoint |

Default config: `data_provider/custom_mindts_anomaly.yaml`

The six MindTS anomaly detection domains:
- **EWJ** (1 channel) — East-West Japan railway logs
- **Energy** (9 channels) — Power grid monitoring
- **Environment** (1 channel) — Environmental sensor data
- **KR** (1 channel) — Korean server metrics
- **MDT** (1 channel) — Machine downtime tracking
- **Weather** (4 channels) — Weather station anomalies

---

## Forecasting (Time-MMD)

Prompt tuning on Time-MMD numerical forecasting datasets:

```bash
CONFIG_PATH=data_provider/custom_timemmd_forecast.yaml \
EXP_NAME=custom_timemmd_forecast_prompt \
PROJECT_NAME=custom_timemmd_forecast_prompt \
CKPT_PATH=./checkpoints/units_x64_pretrain_checkpoint.pth \
bash scripts/custom/run_prompt_all_custom.sh
```

Default config: `data_provider/custom_timemmd_forecast.yaml`

### Benchmark Prediction Lengths

Following the Aurora/Time-MMD benchmark settings:

| Domain | Prediction Lengths |
|--------|--------------------|
| Agriculture, Climate, Economy, Security, SocialGood, Traffic | 6, 8, 10, 12 |
| Energy, Health_US | 12, 24, 36, 48 |
| Environment | 48, 96, 192, 336 |

---

## Original UniTS Tasks

All original UniTS capabilities are fully preserved.

### Multi-Task Learning (Forecasting + Classification)

```bash
# Pretraining + Prompt Learning
bash ./scripts/pretrain_prompt_learning/UniTS_pretrain_x128.sh

# Supervised Learning
bash ./scripts/supervised_learning/UniTS_supervised.sh
```

### Few-Shot Transfer Learning

> **Note:** Follow the instructions in each script to obtain the pretrained checkpoint first.

```bash
# Few-shot on new forecasting/classification tasks
bash ./scripts/few_shot_newdata/UniTS_finetune_few_shot_newdata_pct20.sh
bash ./scripts/few_shot_newdata/UniTS_prompt_tuning_few_shot_newdata_pct20.sh

# Few-shot anomaly detection
bash ./scripts/few_shot_anomaly_detection/UniTS_finetune_few_shot_anomaly_detection.sh
bash ./scripts/few_shot_anomaly_detection/UniTS_prompt_tuning_few_shot_anomaly_detection.sh

# Few-shot imputation
bash ./scripts/few_shot_imputation/UniTS_finetune_few_shot_imputation_mask050.sh
bash ./scripts/few_shot_imputation/UniTS_prompt_tuning_few_shot_imputation_mask050.sh
```

### Zero-Shot Learning

```bash
# Zero-shot on new forecasting lengths
bash ./scripts/zero_shot/UniTS_forecast_new_length_unify.sh

# Zero-shot on new forecasting datasets
bash ./scripts/zero_shot/UniTS_zeroshot_newdata.sh
```

### Using UniTS on Your Own Data

UniTS supports forecasting, classification, imputation, and anomaly detection with a single shared model. See the [Tutorial](Tutorial.md) for a step-by-step guide.

---

## CLI Reference

### Multimodal Text Adapter Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_text_adapter` | flag | `False` | Enable the text cross-attention adapter |
| `--text_emb_dim` | int | `768` | Dimension of precomputed text embeddings |
| `--text_adapter_heads` | int | `4` | Number of attention heads in the cross-attention adapter |
| `--text_adapter_dropout` | float | `0.0` | Dropout rate for the text adapter |
| `--text_gate_init` | float | `-4.0` | Initial logit value for the gating parameter (sigmoid(-4) ≈ 0.018) |
| `--text_gate_type` | str | `scalar` | Gate type: `scalar` or `channel` |
| `--text_warmup_epochs` | int | `0` | Number of epochs to warmup text adapter before prompt tuning |
| `--text_warmup_lr` | float | `None` | Learning rate for text warmup stage (defaults to `--learning_rate`) |

### YAML Dataset Config for Text-Enabled Datasets

Add these fields to your dataset YAML config to enable text:

```yaml
AD_MindTS_EWJ:
  task_name: anomaly_detection
  data: NPY_AD
  root_path: ./dataset/MindTS_AD/EWJ/
  seq_len: 24
  enc_in: 1
  # Text modality fields:
  use_text: true
  text_emb_path_train: train_text_emb.npy
  text_emb_path_test: test_text_emb.npy
```

When `use_text: false` or these fields are omitted, the data loader and model behave exactly like the original UniTS baseline.

---

## Architecture Details

### Text Adapter Modules

Located in [`models/text_adapter.py`](models/text_adapter.py):

| Module | Purpose |
|--------|---------|
| `TextPatchPooler` | Pools timestep-level text embeddings `[B, L, E]` into patch-aligned tokens `[B, N, E]` using masked average pooling. Handles missing/null text gracefully. |
| `TextPatchTokenizer` | Combines `TextPatchPooler` + linear projection to produce `[B, N, D]` text tokens matching UniTS `d_model` dimension. |
| `TextCrossAttentionAdapter` | Cross-attention fusion: `H + sigmoid(gate) * CrossAttn(LN(H), LN(T), LN(T))`. Supports both `[B, V, N, D]` (multivariate) and `[B, N, D]` (univariate) inputs. Gate initialized small to preserve the pretrained model behavior at start. |

### Integration in UniTS

The text adapter is inserted **after the UniTS backbone and before the generation head** in the anomaly detection path (see [`models/UniTS.py`](models/UniTS.py)):

```
Input Window → Patch Embedding → Backbone → [Text Fusion] → Forecast Head → Reconstruction
```

### Training Freeze Rules

| Stage | Trainable | Frozen |
|-------|-----------|--------|
| **Text Warmup** | Text projector, cross-attention adapter, gate | Backbone, patch/position embeddings, forecast head, prompt tokens |
| **Prompt Tuning** | Prompt tokens + text adapter | Backbone, patch/position embeddings, forecast head |
| **Supervised (baseline)** | All parameters | — |

---

## Pretrained Weights

Pretrained weights for the original UniTS models are available at [checkpoints](https://github.com/mims-harvard/UniTS/releases/tag/ckpt).

---

## Acknowledgement

This codebase is built based on the [Time-Series-Library](https://github.com/thuml/Time-Series-Library). The multimodal extension uses datasets from [Time-MMD](https://github.com/AdityaLab/Time-MMD) and [MindTS](https://github.com/AdityaLab/MindTS). Thanks!