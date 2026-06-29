import argparse
import re
from pathlib import Path

import pandas as pd


def parse_seed(path):
    path = Path(path)
    for text in [path.stem, *[part.name for part in path.parents]]:
        match = re.search(r"seed[_-]?(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def read_results(paths):
    frames = []
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        if "seed" not in df.columns:
            df["seed"] = parse_seed(csv_path)
        df["source"] = str(csv_path)
        frames.append(df)
    if not frames:
        raise ValueError("No CSV files were provided.")
    return pd.concat(frames, ignore_index=True)


def select_rows(df, mode):
    if "task_name" in df.columns:
        df = df[df["task_name"] == "long_term_forecast"].copy()
    df = df[df["MSE"].notna() & df["MAE"].notna()].copy()
    if df.empty:
        raise ValueError("No forecasting rows with MSE/MAE found.")
    rows = []
    keys = ["seed", "dataset", "pred_len"]
    for _, group in df.groupby(keys, sort=False):
        if mode == "best":
            idx = group["MSE"].astype(float).idxmin()
        elif mode == "last":
            idx = group["epoch"].astype(int).idxmax()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        rows.append(group.loc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def mean_std_table(selected):
    rows = []
    for (dataset, pred_len), group in selected.groupby(["dataset", "pred_len"], sort=False):
        rows.append({
            "dataset": dataset,
            "pred_len": int(pred_len),
            "seeds": group["seed"].nunique(),
            "MSE_mean": group["MSE"].astype(float).mean(),
            "MSE_std": group["MSE"].astype(float).std(ddof=1),
            "MAE_mean": group["MAE"].astype(float).mean(),
            "MAE_std": group["MAE"].astype(float).std(ddof=1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize UniTS Time-MMD forecasting CSV results.")
    parser.add_argument("csv", nargs="+", help="CSV files, e.g. results/timemmd_forecast/exp/run/seed_*/results_seed*.csv")
    parser.add_argument("--mode", choices=["last", "best"], default="last",
                        help="Use last epoch per seed/dataset/horizon, or best MSE across epochs")
    parser.add_argument("--output", default=None, help="Optional CSV path for selected rows")
    parser.add_argument("--summary-output", default=None, help="Optional CSV path for mean/std table")
    args = parser.parse_args()

    df = read_results(args.csv)
    selected = select_rows(df, args.mode)
    summary = mean_std_table(selected)

    show_cols = ["seed", "dataset", "pred_len", "epoch", "MSE", "MAE"]
    print("\nSelected rows:")
    print(selected[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMean/std over seeds:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output, index=False)
        print(f"\nwrote {output}")
    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
