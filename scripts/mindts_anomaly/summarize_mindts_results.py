import argparse
import re
from pathlib import Path

import pandas as pd


METRIC_COLS = ["Precision", "Recall", "F1", "Aff-F", "V-PR", "V-ROC"]


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
        if "dataset" not in df.columns:
            raise ValueError(f"{csv_path} has no 'dataset' column")
        if "seed" not in df.columns:
            df["seed"] = parse_seed(csv_path)
        df["source"] = str(csv_path)
        frames.append(df)
    if not frames:
        raise ValueError("No CSV files were provided.")
    return pd.concat(frames, ignore_index=True)


def best_rows(df, select_metric):
    if select_metric not in df.columns:
        raise ValueError(f"{select_metric} not found. Columns: {list(df.columns)}")
    rows = []
    for (seed, dataset), group in df.groupby(["seed", "dataset"], sort=False):
        idx = group[select_metric].astype(float).idxmax()
        rows.append(group.loc[idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def mean_std_table(best):
    metric_cols = [col for col in METRIC_COLS if col in best.columns]
    rows = []
    for dataset, group in best.groupby("dataset", sort=False):
        row = {"dataset": dataset, "seeds": group["seed"].nunique()}
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].astype(float).mean()
            row[f"{metric}_std"] = group[metric].astype(float).std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def print_best(best):
    cols = ["seed", "dataset", "epoch", "ratio", *METRIC_COLS]
    cols = [col for col in cols if col in best.columns]
    print("\nBest row per seed/dataset:")
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def print_summary(summary):
    print("\nMean/std over seeds:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main():
    parser = argparse.ArgumentParser(description="Summarize UniTS MindTS CSV results.")
    parser.add_argument("csv", nargs="+", help="CSV files, e.g. results/mindts_prompt/exp/seed_*/mindts_seed*.csv")
    parser.add_argument("--select-metric", default="Aff-F", help="Metric used to select best epoch/ratio row")
    parser.add_argument("--output", default=None, help="Optional CSV path for best rows")
    parser.add_argument("--summary-output", default=None, help="Optional CSV path for mean/std table")
    args = parser.parse_args()

    df = read_results(args.csv)
    best = best_rows(df, args.select_metric)
    summary = mean_std_table(best)

    print_best(best)
    print_summary(summary)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        best.to_csv(output, index=False)
        print(f"\nwrote {output}")

    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
