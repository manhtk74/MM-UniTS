import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TIME_MMD_DEFAULTS = None
MINDTS_AD_DEFAULTS = None

TIME_MMD_PAPER_HORIZONS = {
    "Agriculture": [6, 8, 10, 12],
    "Climate": [6, 8, 10, 12],
    "Economy": [6, 8, 10, 12],
    "Energy": [12, 24, 36, 48],
    "Environment": [48, 96, 192, 336],
    "Health": [12, 24, 36, 48],
    "Health_US": [12, 24, 36, 48],
    "Security": [6, 8, 10, 12],
    "SocialGood": [6, 8, 10, 12],
    "Traffic": [6, 8, 10, 12],
}


def repo_root_from_script():
    return Path(__file__).resolve().parents[3]


def natural_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def find_time_column(df):
    preferred = ["date", "Date", "start_date", "ValidStart", "MapDate", "Month"]
    for name in preferred:
        if name in df.columns:
            return name
    for name in df.columns:
        if "date" in str(name).lower() or "month" in str(name).lower():
            return name
    raise ValueError("Could not find a date-like column")


def parse_dates(values, column_name):
    if column_name == "MapDate":
        parsed = pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed


def clean_time_mmd_domain(repo_root, domain, out_root):
    src = repo_root / "dataset" / "Time-MMD" / "numerical" / domain / f"{domain}.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    date_col = find_time_column(df)
    dates = parse_dates(df[date_col], date_col)

    excluded = {
        "date",
        "date.1",
        "start_date",
        "end_date",
        "month",
        "mapdate",
        "validstart",
        "validend",
        "areaofinterest",
        "statisticformatid",
        "cbsa",
        "cbsa code",
        "category",
        "defining parameter",
        "defining site",
        "region type",
        "region",
        "year",
        "week",
        "year_week",
    }

    numeric = {}
    for col in df.columns:
        if str(col).lower() in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            numeric[col] = values

    if "OT" not in numeric:
        raise ValueError(f"{src} does not contain numeric OT target")

    clean = pd.DataFrame({"date": dates})
    for col, values in numeric.items():
        clean[col] = values
    clean = clean.dropna(subset=["date"]).sort_values("date")
    clean = clean.groupby("date", as_index=False).mean(numeric_only=True)

    value_cols = [col for col in clean.columns if col not in ["date", "OT"]]
    renamed = {"date": "date", "OT": "OT"}
    for idx, col in enumerate(value_cols, start=1):
        renamed[col] = f"value_{idx}"
    clean = clean.rename(columns=renamed)
    ordered = ["date"] + [renamed[col] for col in value_cols] + ["OT"]
    clean = clean[ordered]
    value_only = [col for col in clean.columns if col != "date"]
    clean[value_only] = clean[value_only].interpolate(limit_direction="both").ffill().bfill()
    clean[value_only] = clean[value_only].replace([np.inf, -np.inf], np.nan)
    clean[value_only] = clean[value_only].interpolate(limit_direction="both").ffill().bfill()
    clean = clean.dropna()

    out_dir = out_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{domain}.csv"
    clean.to_csv(out_csv, index=False)

    return {
        "domain": domain,
        "source": str(src),
        "output": str(out_csv),
        "timesteps": int(len(clean)),
        "value_columns": value_only,
        "enc_in_s": 1,
        "enc_in_m": len(value_only),
    }


def read_mindts_meta(repo_root):
    meta_path = repo_root / "dataset" / "MindTS" / "dataset" / "anomaly_detect" / "DETECT_META.csv"
    meta = {}
    if not meta_path.exists():
        return meta
    df = pd.read_csv(meta_path)
    for _, row in df.iterrows():
        file_name = str(row.get("file_name", ""))
        if not file_name.endswith(".csv") or file_name.endswith("_text.csv"):
            continue
        train_lens = row.get("train_lens")
        total_len = row.get("total_len")
        try:
            train_lens = int(float(str(train_lens).strip()))
        except ValueError:
            train_lens = None
        try:
            total_len = int(float(str(total_len).strip()))
        except ValueError:
            total_len = None
        meta[file_name] = {"train_lens": train_lens, "total_len": total_len}
    return meta


def clean_mindts_ad_domain(repo_root, domain, out_root, meta):
    src = repo_root / "dataset" / "MindTS" / "dataset" / "anomaly_detect" / "data" / f"{domain}.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    required = {"date", "data", "cols"}
    if not required.issubset(df.columns):
        raise ValueError(f"{src} must have columns {sorted(required)}")

    df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")
    signal_df = df[df["cols"] != "label"].copy()
    label_df = df[df["cols"] == "label"].copy()

    signal_df["data"] = pd.to_numeric(signal_df["data"], errors="coerce")
    signal_df["data"] = signal_df["data"].replace([np.inf, -np.inf], np.nan)
    pivot = signal_df.pivot_table(index="date", columns="cols", values="data", aggfunc="mean")
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns, key=natural_key), axis=1)
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    pivot = pivot.interpolate(limit_direction="both").ffill().bfill()

    labels = label_df.set_index("date")["data"]
    labels = pd.to_numeric(labels, errors="coerce").reindex(pivot.index).fillna(0).astype(int)

    info = meta.get(f"{domain}.csv", {})
    train_len = info.get("train_lens")
    if train_len is None:
        train_len = int(len(pivot) * 0.8)
    train_len = max(1, min(train_len, len(pivot) - 1))

    train = pivot.iloc[:train_len].to_numpy(dtype=np.float32)
    test = pivot.iloc[train_len:].to_numpy(dtype=np.float32)
    test_label = labels.iloc[train_len:].to_numpy(dtype=np.float32)

    out_dir = out_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "test.npy", test)
    np.save(out_dir / "test_label.npy", test_label)

    anomaly_ratio = float(test_label.mean() * 100.0) if len(test_label) else 0.0
    details = {
        "domain": domain,
        "source": str(src),
        "output": str(out_dir),
        "train_timesteps": int(train.shape[0]),
        "test_timesteps": int(test.shape[0]),
        "channels": int(train.shape[1]),
        "test_anomalies": int(test_label.sum()),
        "anomaly_ratio_percent": anomaly_ratio,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    return details


def discover_time_mmd_domains(repo_root):
    root = repo_root / "dataset" / "Time-MMD" / "numerical"
    domains = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / f"{path.name}.csv").exists():
            domains.append(path.name)
    return domains


def discover_mindts_ad_domains(repo_root):
    root = repo_root / "dataset" / "MindTS" / "dataset" / "anomaly_detect" / "data"
    return sorted(path.stem for path in root.glob("*.csv") if not path.name.endswith("_text.csv"))


def write_yaml(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def timemmd_horizons(domain):
    return TIME_MMD_PAPER_HORIZONS.get(domain, [])


def append_timemmd_forecast_entry(lines, domain, pred_len):
    lines.extend([
        f"  LTF_TimeMMD_{domain}_p{pred_len}:",
        "    task_name: long_term_forecast",
        f"    dataset: TimeMMD_{domain}",
        "    data: custom",
        "    embed: timeF",
        f"    root_path: ./dataset/TimeMMD/{domain}/",
        f"    data_path: {domain}.csv",
        "    features: S",
        "    seq_len: 96",
        "    label_len: 48",
        f"    pred_len: {pred_len}",
        "    enc_in: 1",
        "    dec_in: 1",
        "    c_out: 1",
        "",
    ])


def write_timemmd_pretrain_yaml(units_root, manifest):
    lines = ["task_dataset:"]
    for item in manifest["time_mmd"]:
        domain = item["domain"]
        lines.extend([
            f"  PRE_TimeMMD_{domain}:",
            "    task_name: pretrain_long_term_forecast",
            f"    dataset: TimeMMD_{domain}",
            "    data: custom",
            "    embed: timeF",
            f"    root_path: ./dataset/TimeMMD/{domain}/",
            f"    data_path: {domain}.csv",
            "    features: S",
            "    seq_len: 96",
            "    label_len: 48",
            "    pred_len: 0",
            "    enc_in: 1",
            "    dec_in: 1",
            "    c_out: 1",
            "",
        ])
    write_yaml(units_root / "data_provider" / "custom_timemmd_pretrain.yaml", lines)


def write_timemmd_forecast_yaml(units_root, manifest):
    lines = ["task_dataset:"]
    for item in manifest["time_mmd"]:
        domain = item["domain"]
        for pred_len in timemmd_horizons(domain):
            append_timemmd_forecast_entry(lines, domain, pred_len)
    write_yaml(units_root / "data_provider" / "custom_timemmd_forecast.yaml", lines)


def write_mindts_anomaly_yaml(units_root, manifest):
    lines = ["task_dataset:"]
    for item in manifest["mindts_ad"]:
        domain = item["domain"]
        channels = item["channels"]
        lines.extend([
            f"  AD_MindTS_{domain}:",
            "    task_name: anomaly_detection",
            f"    dataset_name: MindTS_{domain}",
            f"    dataset: MindTS_{domain}",
            "    data: NPY_AD",
            f"    root_path: ./dataset/MindTS_AD/{domain}/",
            "    seq_len: 24",
            "    label_len: 0",
            "    pred_len: 0",
            "    features: M",
            "    embed: timeF",
            f"    enc_in: {channels}",
            f"    dec_in: {channels}",
            f"    c_out: {channels}",
            "",
        ])
    write_yaml(units_root / "data_provider" / "custom_mindts_anomaly.yaml", lines)


def append_timemmd_forecast_entries(lines, manifest):
    for item in manifest["time_mmd"]:
        domain = item["domain"]
        for pred_len in timemmd_horizons(domain):
            append_timemmd_forecast_entry(lines, domain, pred_len)


def append_mindts_anomaly_entries(lines, manifest):
    for item in manifest["mindts_ad"]:
        domain = item["domain"]
        channels = item["channels"]
        lines.extend([
            f"  AD_MindTS_{domain}:",
            "    task_name: anomaly_detection",
            f"    dataset_name: MindTS_{domain}",
            f"    dataset: MindTS_{domain}",
            "    data: NPY_AD",
            f"    root_path: ./dataset/MindTS_AD/{domain}/",
            "    seq_len: 24",
            "    label_len: 0",
            "    pred_len: 0",
            "    features: M",
            "    embed: timeF",
            f"    enc_in: {channels}",
            f"    dec_in: {channels}",
            f"    c_out: {channels}",
            "",
        ])


def write_all_prompt_yaml(units_root, manifest):
    lines = ["task_dataset:"]
    append_timemmd_forecast_entries(lines, manifest)
    append_mindts_anomaly_entries(lines, manifest)
    write_yaml(units_root / "data_provider" / "custom_all_prompt.yaml", lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--timemmd-domains", nargs="*", default=TIME_MMD_DEFAULTS)
    parser.add_argument("--mindts-ad-domains", nargs="*", default=MINDTS_AD_DEFAULTS)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    units_root = repo_root / "UniTS"
    timemmd_domains = args.timemmd_domains or discover_time_mmd_domains(repo_root)
    mindts_ad_domains = args.mindts_ad_domains or discover_mindts_ad_domains(repo_root)

    timemmd_out = units_root / "dataset" / "TimeMMD"
    mindts_out = units_root / "dataset" / "MindTS_AD"
    timemmd_out.mkdir(parents=True, exist_ok=True)
    mindts_out.mkdir(parents=True, exist_ok=True)

    manifest = {"time_mmd": [], "mindts_ad": []}
    for domain in timemmd_domains:
        manifest["time_mmd"].append(clean_time_mmd_domain(repo_root, domain, timemmd_out))

    meta = read_mindts_meta(repo_root)
    for domain in mindts_ad_domains:
        manifest["mindts_ad"].append(clean_mindts_ad_domain(repo_root, domain, mindts_out, meta))

    write_timemmd_pretrain_yaml(units_root, manifest)
    write_timemmd_forecast_yaml(units_root, manifest)
    write_mindts_anomaly_yaml(units_root, manifest)
    write_all_prompt_yaml(units_root, manifest)

    manifest_path = units_root / "dataset" / "custom_data_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
