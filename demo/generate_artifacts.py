"""Generate offline SMD artifacts for the Streamlit monitoring dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.alert_engine import build_incidents


SEQ_LEN = 96
LABEL_LEN = 48
HORIZON = 24
FORECAST_STRIDE = 24
ROLLING_WINDOW = 12
WARMUP = 96


def causal_rolling_median(values: np.ndarray, history: np.ndarray | None = None,
                          window: int = ROLLING_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if history is None:
        context = values
        offset = 0
    else:
        history = np.asarray(history, dtype=np.float32)[-window:]
        context = np.concatenate([history, values], axis=0)
        offset = len(history)
    frame = pd.DataFrame(context)
    expected = frame.rolling(window=window, min_periods=1).median().shift(1)
    expected.iloc[0] = frame.iloc[0]
    expected = expected.ffill().to_numpy(dtype=np.float32)
    return expected[offset:]


def anomaly_metrics(labels, predictions, scores, extended=False):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics = {
        "Precision": float(precision * 100.0),
        "Recall": float(recall * 100.0),
        "F1": float(f1 * 100.0),
        "Aff-F": None,
        "V-PR": None,
        "V-ROC": None,
    }
    if extended:
        try:
            from utils.mindts_metrics import affiliation_f, vus_pr, vus_roc

            metrics.update({
                "Aff-F": float(affiliation_f(labels, predictions, scores) * 100.0),
                "V-PR": float(vus_pr(labels, predictions, scores) * 100.0),
                "V-ROC": float(vus_roc(labels, predictions, scores) * 100.0),
            })
        except Exception as exc:  # extended metrics should not block the demo
            print(f"warning: extended metrics failed: {exc}")
    return metrics


def forecast_metrics(test: np.ndarray, origins: np.ndarray, forecasts: np.ndarray) -> dict:
    if len(origins) == 0:
        return {"MAE": None, "MSE": None}
    targets = np.stack([
        test[origin + 1:origin + 1 + forecasts.shape[1]] for origin in origins
    ])
    error = forecasts.astype(np.float64) - targets.astype(np.float64)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(np.square(error))),
    }


def baseline_outputs(train, test, scale):
    train_expected = causal_rolling_median(train)
    train_residual = (train - train_expected) / scale
    train_scores = np.mean(np.square(train_residual), axis=1)
    threshold = float(np.percentile(train_scores[WARMUP:], 99.0))

    expected = causal_rolling_median(test, history=train)
    residual = (test - expected) / scale
    scores = np.mean(np.square(residual), axis=1)
    scores[:WARMUP] = 0.0
    origins = np.arange(SEQ_LEN - 1, len(test) - HORIZON, FORECAST_STRIDE, dtype=np.int32)
    forecasts = np.repeat(test[origins, None, :], HORIZON, axis=1).astype(np.float32)
    return expected, residual, scores, threshold, origins, forecasts


def load_model(checkpoint: Path, config_path: Path, args):
    import torch
    import yaml
    from models import UniTS

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["task_dataset"]
    config_list = [[name, values] for name, values in config.items()]
    model_args = SimpleNamespace(
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        prompt_num=args.prompt_num,
        patch_len=args.patch_len,
        stride=args.stride,
        dropout=0.0,
        use_text_modality=False,
    )
    model = UniTS.Model(model_args, config_list)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("student", "state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    normalized = {}
    target_state = model.state_dict()
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key in target_state and target_state[key].shape == value.shape:
            normalized[key] = value
    message = model.load_state_dict(normalized, strict=False)
    print(
        f"Loaded {len(normalized)} tensors from {checkpoint}; "
        f"missing={len(message.missing_keys)}, unexpected={len(message.unexpected_keys)}"
    )
    task_ids = {name: index for index, (name, _) in enumerate(config_list)}
    device = torch.device(args.device)
    model.to(device).eval()
    return model, task_ids, device


def _sliding_last_predictions(model, values, task_id, task_name, device, batch_size):
    import torch

    tensor = torch.from_numpy(values.astype(np.float32))
    windows = tensor.unfold(0, SEQ_LEN, 1).permute(0, 2, 1)
    predictions = np.empty_like(values, dtype=np.float32)
    predictions[:SEQ_LEN - 1] = values[:SEQ_LEN - 1]
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size].to(device)
            output = model(
                batch, None, None, None,
                task_id=task_id,
                task_name=task_name,
            )
            predictions[SEQ_LEN - 1 + start:SEQ_LEN - 1 + start + len(batch)] = (
                output[:, -1].detach().cpu().numpy()
            )
    return predictions


def units_outputs(model, task_ids, device, train, test, mean, scale, batch_size):
    import torch

    train_norm = ((train - mean) / scale).astype(np.float32)
    test_norm = ((test - mean) / scale).astype(np.float32)
    train_expected = _sliding_last_predictions(
        model, train_norm, task_ids["AD_SMD"], "anomaly_detection", device, batch_size
    )
    train_scores = np.mean(np.square(train_norm - train_expected), axis=1)
    threshold = float(np.percentile(train_scores[WARMUP:], 99.0))

    expected_norm = _sliding_last_predictions(
        model, test_norm, task_ids["AD_SMD"], "anomaly_detection", device, batch_size
    )
    residual = test_norm - expected_norm
    scores = np.mean(np.square(residual), axis=1)
    scores[:WARMUP] = 0.0
    expected = expected_norm * scale + mean

    origins = np.arange(SEQ_LEN - 1, len(test) - HORIZON, FORECAST_STRIDE, dtype=np.int32)
    forecasts = []
    with torch.inference_mode():
        for start in range(0, len(origins), batch_size):
            batch_origins = origins[start:start + batch_size]
            windows = np.stack([
                test_norm[origin - SEQ_LEN + 1:origin + 1] for origin in batch_origins
            ])
            output = model(
                torch.from_numpy(windows).to(device), None, None, None,
                task_id=task_ids["LTF_SMD_p24"],
                task_name="long_term_forecast",
            )
            forecasts.append(output[:, -HORIZON:].detach().cpu().numpy())
    forecast_values = np.concatenate(forecasts, axis=0) if forecasts else np.empty((0, HORIZON, 38))
    forecast_values = forecast_values * scale + mean
    return (
        expected.astype(np.float32), residual, scores, threshold,
        origins, forecast_values.astype(np.float32),
    )


def write_machine_artifact(output_root, machine_id, backend, train, test, labels,
                           interpretation, mean, scale, args, model_context=None):
    if backend == "baseline":
        expected, residual, scores, threshold, origins, forecasts = baseline_outputs(
            train, test, scale
        )
    else:
        model, task_ids, device = model_context
        expected, residual, scores, threshold, origins, forecasts = units_outputs(
            model, task_ids, device, train, test, mean, scale, args.batch_size
        )

    predictions = (scores > threshold).astype(np.uint8)
    incidents = build_incidents(scores, threshold, residual, warmup=WARMUP)
    metrics = {
        "backend": backend,
        "machine_id": machine_id,
        "threshold": threshold,
        "threshold_protocol": "train_percentile_99",
        "anomaly": anomaly_metrics(labels, predictions, scores, args.extended_metrics),
        "forecast": forecast_metrics(test, origins, forecasts),
    }
    directory = output_root / machine_id
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "series.npz",
        actual=test.astype(np.float32),
        expected=expected.astype(np.float32),
        anomaly_score=scores.astype(np.float32),
        predicted_label=predictions,
        ground_truth=labels.astype(np.uint8),
        forecast_origins=origins,
        forecasts=forecasts.astype(np.float32),
        threshold=np.float32(threshold),
    )
    (directory / "incidents.json").write_text(
        json.dumps({
            "incidents": incidents,
            "ground_truth_interpretation": interpretation,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{machine_id}: threshold={threshold:.5f}, incidents={len(incidents)}, "
        f"F1={metrics['anomaly']['F1']:.2f}, MAE={metrics['forecast']['MAE']:.5f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Generate offline SMD dashboard artifacts")
    parser.add_argument("--backend", choices=("baseline", "units"), default="baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/SMD"))
    parser.add_argument("--output-dir", type=Path, default=Path("demo/artifacts"))
    parser.add_argument("--machine", action="append", dest="machines")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("data_provider/smd_demo.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extended-metrics", action="store_true")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--prompt-num", type=int, default=10)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    return parser


def main():
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "scaler.npz") as scaler:
        mean = scaler["mean"].astype(np.float32)
        scale = scaler["scale"].astype(np.float32)
    machine_ids = args.machines or list(manifest["machines"].keys())
    unknown = sorted(set(machine_ids) - set(manifest["machines"]))
    if unknown:
        raise ValueError(f"Unknown machines: {unknown}")

    model_context = None
    if args.backend == "units":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for the units backend")
        model_context = load_model(args.checkpoint, args.config, args)

    output_root = args.output_dir.resolve() / args.backend
    output_root.mkdir(parents=True, exist_ok=True)
    for machine_id in machine_ids:
        with np.load(data_dir / "machines" / f"{machine_id}.npz") as values:
            train = values["train"].astype(np.float32)
            test = values["test"].astype(np.float32)
            labels = values["label"].astype(np.uint8)
        write_machine_artifact(
            output_root, machine_id, args.backend, train, test, labels,
            manifest["machines"][machine_id]["interpretation"],
            mean, scale, args, model_context,
        )

    backend_manifest = {
        "schema_version": 1,
        "backend": args.backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machines": machine_ids,
        "feature_count": 38,
        "seq_len": SEQ_LEN,
        "forecast_horizon": HORIZON,
        "forecast_stride": FORECAST_STRIDE,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "config": str(args.config),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(backend_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
