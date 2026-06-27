import argparse
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import adjustment


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(args, task_data_config_list, device):
    import importlib

    module = importlib.import_module("models." + args.model)
    model = module.Model(args, task_data_config_list).to(device)

    ckpt = load_checkpoint(args.pretrained_weight)
    if isinstance(ckpt, dict) and "student" in ckpt:
        ckpt = {k: v for k, v in ckpt["student"].items() if "cls_prompts" not in k}

    msg = model.load_state_dict(ckpt, strict=False)
    print("[checkpoint]", args.pretrained_weight)
    print("[load_state_dict]", msg)
    model.eval()
    return model


def read_single_task_config(config_path, dataset_key):
    with open(config_path, "r", encoding="utf-8") as f:
        task_dataset = yaml.load(f, Loader=yaml.FullLoader)["task_dataset"]

    if dataset_key in task_dataset:
        return dataset_key, task_dataset[dataset_key]

    dataset_key_lower = dataset_key.lower()
    matches = []
    for name, config in task_dataset.items():
        aliases = {
            name.lower(),
            str(config.get("dataset", "")).lower(),
            str(config.get("dataset_name", "")).lower(),
        }
        if dataset_key_lower in aliases:
            matches.append((name, config))

    if len(matches) != 1:
        available = ", ".join(task_dataset.keys())
        raise ValueError(
            f"Could not uniquely resolve dataset_key={dataset_key!r}. Available: {available}"
        )

    return matches[0]


def make_runtime_args(cli_args):
    return SimpleNamespace(
        model=cli_args.model,
        pretrained_weight=cli_args.pretrained_weight,
        prompt_num=cli_args.prompt_num,
        patch_len=cli_args.patch_len,
        stride=cli_args.stride,
        e_layers=cli_args.e_layers,
        d_model=cli_args.d_model,
        n_heads=cli_args.n_heads,
        dropout=cli_args.dropout,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        subsample_pct=None,
        fix_seed=cli_args.seed,
        target=cli_args.target,
        freq=cli_args.freq,
        inverse=cli_args.inverse,
        features="M",
    )


def autocast_context(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def maybe_limit_batches(iterator, max_batches):
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def export_forecast(args, model, task_name, task_config, device):
    task_data_config_list = [[task_name, task_config]]
    test_data, test_loader = data_provider(args, task_config, flag="test", ddp=False)
    pred_len = task_config["pred_len"]
    features = task_config["features"]
    f_dim = -1 if features == "MS" else 0

    preds = []
    trues = []

    with torch.no_grad():
        for _, (batch_x, batch_y, _, _) in maybe_limit_batches(test_loader, args.max_batches):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            with autocast_context(device):
                outputs = model(
                    batch_x,
                    None,
                    None,
                    None,
                    task_id=0,
                    task_name="long_term_forecast",
                )

            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:]

            outputs = outputs.detach().cpu()
            batch_y = batch_y.detach().cpu()

            if getattr(test_data, "scale", False) and args.inverse:
                outputs = test_data.inverse_transform(outputs)
                batch_y = test_data.inverse_transform(batch_y)

            preds.append(np.asarray(outputs, dtype=np.float32))
            trues.append(np.asarray(batch_y, dtype=np.float32))

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe = metric(y_pred, y_true)

    metadata = {
        "task": "forecast",
        "dataset_key": task_name,
        "dataset": task_config.get("dataset"),
        "seq_len": task_config.get("seq_len"),
        "label_len": task_config.get("label_len"),
        "pred_len": pred_len,
        "features": features,
        "seed": args.seed,
        "checkpoint": args.pretrained_weight,
        "max_batches": args.max_batches,
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "shape_y_true": list(y_true.shape),
        "shape_y_pred": list(y_pred.shape),
    }

    return {"y_true": y_true, "y_pred": y_pred}, metadata


def compute_ad_scores(args, model, loader, device, include_labels):
    criterion = nn.MSELoss(reduction="none")
    scores = []
    labels = []

    with torch.no_grad():
        for _, (batch_x, batch_y) in maybe_limit_batches(loader, args.max_batches):
            batch_x = batch_x.float().to(device)
            with autocast_context(device):
                outputs = model(
                    batch_x,
                    None,
                    None,
                    None,
                    task_id=0,
                    task_name="anomaly_detection",
                )
            score = torch.mean(criterion(batch_x, outputs), dim=-1)
            scores.append(score.detach().cpu().numpy().astype(np.float32))
            if include_labels:
                labels.append(np.asarray(batch_y, dtype=np.float32))

    flat_scores = np.concatenate(scores, axis=0).reshape(-1).astype(np.float32)
    if not include_labels:
        return flat_scores, None

    flat_labels = np.concatenate(labels, axis=0).reshape(-1).astype(np.int8)
    return flat_scores, flat_labels


def export_anomaly(args, model, task_name, task_config, device):
    train_data, train_loader = data_provider(args, task_config, flag="train", ddp=False)
    test_data, test_loader = data_provider(args, task_config, flag="test", ddp=False)

    train_scores, _ = compute_ad_scores(args, model, train_loader, device, include_labels=False)
    test_scores, test_labels = compute_ad_scores(args, model, test_loader, device, include_labels=True)

    if args.max_batches is None:
        combined = np.concatenate([train_scores, test_scores], axis=0)
        threshold = np.percentile(combined, 100 - args.anomaly_ratio)
        pred = (test_scores > threshold).astype(np.int8)
        gt = test_labels.astype(np.int8)
        gt_adjusted, pred_adjusted = adjustment(gt.copy(), pred.copy())
        accuracy = float((pred_adjusted == gt_adjusted).mean())
    else:
        threshold = np.nan
        pred = np.zeros_like(test_labels, dtype=np.int8)
        gt_adjusted = test_labels.astype(np.int8)
        pred_adjusted = pred
        accuracy = np.nan

    metadata = {
        "task": "anomaly_detection",
        "dataset_key": task_name,
        "dataset": task_config.get("dataset"),
        "seq_len": task_config.get("seq_len"),
        "seed": args.seed,
        "checkpoint": args.pretrained_weight,
        "anomaly_ratio": args.anomaly_ratio,
        "max_batches": args.max_batches,
        "original_threshold": float(threshold),
        "original_adjusted_accuracy": float(accuracy),
        "shape_train_scores": list(train_scores.shape),
        "shape_test_scores": list(test_scores.shape),
        "shape_test_labels": list(test_labels.shape),
    }

    arrays = {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "original_pred": pred.astype(np.int8),
        "original_pred_adjusted": pred_adjusted.astype(np.int8),
        "original_gt_adjusted": gt_adjusted.astype(np.int8),
    }
    return arrays, metadata


def save_npz(output_path, arrays, metadata):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata=np.array(json.dumps(metadata, indent=2), dtype=object),
    )
    print("[saved]", output_path)
    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export original UniTS point/reconstruction artifacts as .npz."
    )
    parser.add_argument("--task", choices=["forecast", "ad"], required=True)
    parser.add_argument("--dataset_key", required=True)
    parser.add_argument("--task_data_config_path", required=True)
    parser.add_argument("--pretrained_weight", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="UniTS")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--anomaly_ratio", type=float, default=1.0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    device = torch.device(
        "cuda"
        if cli_args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if cli_args.device == "auto"
        else cli_args.device
    )
    print("[device]", device)

    runtime_args = make_runtime_args(cli_args)
    task_name, task_config = read_single_task_config(
        cli_args.task_data_config_path, cli_args.dataset_key
    )
    task_data_config_list = [[task_name, task_config]]
    model = load_model(runtime_args, task_data_config_list, device)

    # Attach export-only options after model construction.
    for key, value in vars(cli_args).items():
        setattr(runtime_args, key, value)

    if cli_args.task == "forecast":
        arrays, metadata = export_forecast(runtime_args, model, task_name, task_config, device)
    else:
        arrays, metadata = export_anomaly(runtime_args, model, task_name, task_config, device)

    save_npz(cli_args.output_path, arrays, metadata)


if __name__ == "__main__":
    main()
