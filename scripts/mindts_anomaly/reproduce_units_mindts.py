import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.UniTS import Model as UniTSModel
from utils.mindts_metrics import affiliation_f, vus_pr, vus_roc


DEFAULT_DATASETS = ["Weather", "Energy", "Environment", "KR", "EWJ", "MDT"]
DEFAULT_RATIOS = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]

PAPER_UNITS_TABLE1 = {
    "Weather": {"Precision": 35.88, "Recall": 82.46, "F1": 50.00, "Aff-F": 76.17, "V-PR": 44.35, "V-ROC": 75.08},
    "Energy": {"Precision": 20.20, "Recall": 73.21, "F1": 31.66, "Aff-F": 63.84, "V-PR": 31.04, "V-ROC": 51.15},
    "Environment": {"Precision": 35.96, "Recall": 83.33, "F1": 50.24, "Aff-F": 83.06, "V-PR": 50.24, "V-ROC": 92.03},
    "KR": {"Precision": 30.23, "Recall": 79.79, "F1": 43.84, "Aff-F": 82.24, "V-PR": 43.32, "V-ROC": 73.93},
    "EWJ": {"Precision": 26.95, "Recall": 71.70, "F1": 39.18, "Aff-F": 77.61, "V-PR": 39.32, "V-ROC": 73.91},
    "MDT": {"Precision": 44.19, "Recall": 62.30, "F1": 51.70, "Aff-F": 75.57, "V-PR": 37.61, "V-ROC": 58.67},
}


class WindowDataset(Dataset):
    def __init__(self, data, labels=None, win_size=96, step=1):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.win_size = win_size
        self.step = step

    def __len__(self):
        if len(self.data) < self.win_size:
            return 0
        return (len(self.data) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        start = index * self.step
        end = start + self.win_size
        x = self.data[start:end]
        if self.labels is None:
            y = np.zeros((self.win_size,), dtype=np.float32)
        else:
            y = self.labels[start:end]
        return x, y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_mindts_otb_csv(path):
    raw = pd.read_csv(path)
    label_exists = "label" in raw["cols"].values
    points_per_col = raw.iloc[:, 2].value_counts().max()
    col_names = list(raw["cols"].unique())
    values = {
        col_names[i]: raw.iloc[i * points_per_col : (i + 1) * points_per_col, 1].to_numpy()
        for i in range(len(col_names))
    }
    df = pd.DataFrame(values)
    if label_exists:
        df = df.rename(columns={df.columns[-1]: "label"})
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_mindts_dataset(mindts_root, dataset_name):
    csv_name = dataset_name if dataset_name.endswith(".csv") else f"{dataset_name}.csv"
    mindts_root = Path(mindts_root)
    flat_data_path = mindts_root / csv_name
    flat_meta_path = mindts_root / "DETECT_META.csv"
    nested_data_path = mindts_root / "dataset" / "anomaly_detect" / "data" / csv_name
    nested_meta_path = mindts_root / "dataset" / "anomaly_detect" / "DETECT_META.csv"

    if flat_data_path.exists() and flat_meta_path.exists():
        data_path = flat_data_path
        meta_path = flat_meta_path
    else:
        data_path = nested_data_path
        meta_path = nested_meta_path

    if not data_path.exists():
        raise FileNotFoundError(f"MindTS data file not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"MindTS metadata file not found: {meta_path}")

    df = read_mindts_otb_csv(data_path)
    meta = pd.read_csv(meta_path).set_index("file_name")
    train_len = int(meta.loc[csv_name, "train_lens"])
    features = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    return features[:train_len], labels[:train_len], features[train_len:], labels[train_len:]


def build_args(cli_args):
    return SimpleNamespace(
        prompt_num=cli_args.prompt_num,
        patch_len=cli_args.patch_len,
        stride=cli_args.stride,
        e_layers=cli_args.e_layers,
        d_model=cli_args.d_model,
        n_heads=cli_args.n_heads,
        dropout=cli_args.dropout,
    )


def build_config(dataset_name, enc_in, seq_len):
    dataset_key = f"MindTS_{dataset_name}"
    return [
        [
            dataset_name,
            {
                "task_name": "anomaly_detection",
                "dataset_name": dataset_name,
                "dataset": dataset_key,
                "data": dataset_name,
                "root_path": "",
                "seq_len": seq_len,
                "label_len": 0,
                "pred_len": 0,
                "features": "M",
                "embed": "timeF",
                "enc_in": enc_in,
                "dec_in": enc_in,
                "c_out": enc_in,
            },
        ]
    ]


def normalize_state_dict(state):
    if isinstance(state, dict) and "student" in state:
        state = state["student"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format.")
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def compatible_state_dict(model, state):
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        compatible[key] = value
    return compatible, skipped


def load_pretrained(model, checkpoint_path, device):
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    state = normalize_state_dict(state)
    state, skipped = compatible_state_dict(model, state)
    msg = model.load_state_dict(state, strict=False)
    return {"load_msg": msg, "skipped_size_mismatch": skipped}


def make_train_loader(train_data, seq_len, batch_size, seed, subsample_pct):
    dataset = WindowDataset(train_data, win_size=seq_len, step=1)
    if len(dataset) == 0:
        raise ValueError(f"Training split is shorter than seq_len={seq_len}.")
    if subsample_pct is not None and subsample_pct < 1.0:
        rng = np.random.default_rng(seed)
        keep = max(1, int(len(dataset) * subsample_pct))
        indices = rng.permutation(len(dataset))[:keep]
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def make_eval_loader(data, labels, seq_len, batch_size, step):
    dataset = WindowDataset(data, labels=labels, win_size=seq_len, step=step)
    if len(dataset) == 0:
        raise ValueError(f"Evaluation split is shorter than seq_len={seq_len}.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def train_one_dataset(model, train_loader, epochs, lr, weight_decay, acc_it, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (batch_x, _) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device)
            output = model(batch_x, None, None, None, task_id=0, task_name="anomaly_detection")
            loss = criterion(output, batch_x) / acc_it
            loss.backward()
            if step % acc_it == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * acc_it * len(batch_x)
            total_items += len(batch_x)
        if len(train_loader) % acc_it != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(f"epoch={epoch + 1}/{epochs} train_loss={total_loss / max(total_items, 1):.6f}")


@torch.no_grad()
def reconstruction_scores(model, loader, device):
    criterion = nn.MSELoss(reduction="none")
    scores = []
    labels = []
    model.eval()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        output = model(batch_x, None, None, None, task_id=0, task_name="anomaly_detection")
        score = torch.mean(criterion(output, batch_x), dim=-1)
        scores.append(score.cpu().numpy())
        labels.append(batch_y.numpy())
    return np.concatenate(scores, axis=0).reshape(-1), np.concatenate(labels, axis=0).reshape(-1)


def pad_to_length(values, length):
    values = np.asarray(values)
    if len(values) >= length:
        return values[:length]
    return np.pad(values, (0, length - len(values)), mode="constant", constant_values=0)


def evaluate_with_mindts_metrics(labels, scores, ratios, train_scores, threshold_test_scores, metrics):
    affiliation_f, vus_pr, vus_roc = metrics
    combined = np.concatenate([train_scores, threshold_test_scores])
    rows = []
    labels = labels.astype(float)
    scores = pad_to_length(scores, len(labels)).astype(float)
    for ratio in ratios:
        threshold = np.percentile(combined, 100.0 - ratio)
        pred = pad_to_length((scores > threshold).astype(float), len(labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.astype(int),
            pred.astype(int),
            average="binary",
            zero_division=0,
        )
        aff_f = np.nan_to_num(affiliation_f(labels, pred, scores) * 100.0, nan=0.0)
        v_pr = np.nan_to_num(vus_pr(labels, pred, scores) * 100.0, nan=0.0)
        v_roc = np.nan_to_num(vus_roc(labels, pred, scores) * 100.0, nan=0.0)
        rows.append(
            {
                "ratio": ratio,
                "threshold": threshold,
                "Precision": precision * 100.0,
                "Recall": recall * 100.0,
                "F1": f1 * 100.0,
                "Aff-F": aff_f,
                "V-PR": v_pr,
                "V-ROC": v_roc,
            }
        )
    return rows


def run_dataset(dataset_name, cli_args, metrics):
    print(f"\n=== {dataset_name} ===")
    train_x, _, test_x, test_y = load_mindts_dataset(cli_args.mindts_root, dataset_name)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    model_args = build_args(cli_args)
    configs = build_config(dataset_name, train_x.shape[1], cli_args.seq_len)
    model = UniTSModel(model_args, configs).to(cli_args.device)
    load_msg = load_pretrained(model, cli_args.checkpoint, cli_args.device) if cli_args.checkpoint else None
    if load_msg is not None:
        skipped = load_msg["skipped_size_mismatch"]
        print(f"loaded checkpoint: {load_msg['load_msg']}")
        if skipped:
            preview = ", ".join(item[0] for item in skipped[:5])
            print(f"skipped {len(skipped)} size-mismatched tensors: {preview}")

    train_loader = make_train_loader(
        train_x,
        seq_len=cli_args.seq_len,
        batch_size=cli_args.batch_size,
        seed=cli_args.seed,
        subsample_pct=cli_args.subsample_pct,
    )
    if cli_args.epochs > 0:
        train_one_dataset(
            model,
            train_loader,
            epochs=cli_args.epochs,
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
            acc_it=cli_args.acc_it,
            device=cli_args.device,
        )

    threshold_test_loader = make_eval_loader(test_x, test_y, cli_args.seq_len, cli_args.batch_size, step=1)
    final_step = 1 if cli_args.score_mode == "overlap" else cli_args.seq_len
    final_test_loader = make_eval_loader(test_x, test_y, cli_args.seq_len, cli_args.batch_size, step=final_step)

    train_scores, _ = reconstruction_scores(model, make_eval_loader(train_x, None, cli_args.seq_len, cli_args.batch_size, step=1), cli_args.device)
    threshold_test_scores, _ = reconstruction_scores(model, threshold_test_loader, cli_args.device)
    final_scores, _ = reconstruction_scores(model, final_test_loader, cli_args.device)
    metric_rows = evaluate_with_mindts_metrics(
        labels=test_y,
        scores=final_scores,
        ratios=cli_args.anomaly_ratios,
        train_scores=train_scores,
        threshold_test_scores=threshold_test_scores,
        metrics=metrics,
    )
    best = max(metric_rows, key=lambda row: np.nan_to_num(row["Aff-F"], nan=-1.0))
    paper = PAPER_UNITS_TABLE1.get(dataset_name, {})
    print(
        "best_ratio={ratio} P={Precision:.2f} R={Recall:.2f} F1={F1:.2f} "
        "Aff-F={Aff-F:.2f} V-PR={V-PR:.2f} V-ROC={V-ROC:.2f}".format(
            **best
        )
    )
    if paper:
        print(
            "paper UniTS P={Precision:.2f} R={Recall:.2f} F1={F1:.2f} "
            "Aff-F={Aff-F:.2f} V-PR={V-PR:.2f} V-ROC={V-ROC:.2f}".format(
                **paper
            )
        )
    for row in metric_rows:
        row["dataset"] = dataset_name
        row["paper_Precision"] = paper.get("Precision", math.nan)
        row["paper_Recall"] = paper.get("Recall", math.nan)
        row["paper_F1"] = paper.get("F1", math.nan)
        row["paper_Aff-F"] = paper.get("Aff-F", math.nan)
        row["paper_V-PR"] = paper.get("V-PR", math.nan)
        row["paper_V-ROC"] = paper.get("V-ROC", math.nan)
    return metric_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce UniTS anomaly detection on MindTS datasets.")
    parser.add_argument("--mindts-root", default=str(ROOT / "dataset" / "MindTS"), help="Path to the local MindTS dataset folder. The default is UniTS/dataset/MindTS.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--checkpoint", default=None, help="Path to UniTS pretrained checkpoint, e.g. newcheckpoints/units_x32_pretrain_checkpoint.pth")
    parser.add_argument("--output", default="results/mindts_units_anomaly.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--prompt-num", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--acc-it", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--subsample-pct", type=float, default=0.05)
    parser.add_argument("--score-mode", choices=["thre", "overlap"], default="thre", help="`thre` matches the MindTS baseline adapter's non-overlap final scoring.")
    parser.add_argument("--anomaly-ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device = torch.device(args.device)
    metrics = (affiliation_f, vus_pr, vus_roc)
    all_rows = []
    for dataset_name in args.datasets:
        all_rows.extend(run_dataset(dataset_name, args, metrics))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "ratio",
        "threshold",
        "Precision",
        "Recall",
        "F1",
        "Aff-F",
        "V-PR",
        "V-ROC",
        "paper_Precision",
        "paper_Recall",
        "paper_F1",
        "paper_Aff-F",
        "paper_V-PR",
        "paper_V-ROC",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {output}")

    if args.save_json:
        with Path(args.save_json).open("w") as f:
            json.dump(all_rows, f, indent=2)


if __name__ == "__main__":
    main()
