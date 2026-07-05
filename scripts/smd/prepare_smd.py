"""Validate and preprocess the local Server Machine Dataset (SMD).

This script deliberately has no download path.  It converts the four folders
from the OmniAnomaly release into files consumed by UniTS and the demo while
leaving the source directory untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


EXPECTED_MACHINES = 28
EXPECTED_FEATURES = 38
PARTS = ("train", "test", "test_label", "interpretation_label")
MACHINE_RE = re.compile(r"machine-(\d+)-(\d+)\.txt$")
INTERVAL_RE = re.compile(r"^(\d+)-(\d+):([\d,]+)$")


def machine_sort_key(path_or_name: Path | str) -> tuple[int, int]:
    name = Path(path_or_name).name
    match = MACHINE_RE.match(name)
    if not match:
        raise ValueError(f"Invalid SMD machine filename: {name}")
    return int(match.group(1)), int(match.group(2))


def parse_interpretation(path: Path) -> list[dict]:
    """Parse ``start-end:dimension,...`` records (dimensions remain 1-based)."""
    intervals: list[dict] = []
    tokens = path.read_text(encoding="utf-8").split()
    for token in tokens:
        match = INTERVAL_RE.match(token)
        if not match:
            raise ValueError(f"Invalid interpretation token in {path}: {token}")
        start, end = int(match.group(1)), int(match.group(2))
        dimensions = [int(value) for value in match.group(3).split(",")]
        if start > end:
            raise ValueError(f"Invalid interval {start}-{end} in {path}")
        if any(value < 1 or value > EXPECTED_FEATURES for value in dimensions):
            raise ValueError(f"Dimension outside 1..{EXPECTED_FEATURES} in {path}")
        intervals.append({"start": start, "end": end, "dimensions": dimensions})
    return intervals


def discover_files(raw_dir: Path) -> dict[str, dict[str, Path]]:
    discovered: dict[str, dict[str, Path]] = {}
    expected_names: set[str] | None = None
    for part in PARTS:
        directory = raw_dir / part
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing SMD directory: {directory}")
        files = sorted(directory.glob("machine-*.txt"), key=machine_sort_key)
        names = {path.stem for path in files}
        if len(files) != EXPECTED_MACHINES:
            raise ValueError(
                f"Expected {EXPECTED_MACHINES} files in {directory}, found {len(files)}"
            )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            missing = sorted(expected_names - names)
            extra = sorted(names - expected_names)
            raise ValueError(f"Machine mismatch in {directory}: missing={missing}, extra={extra}")
        for path in files:
            discovered.setdefault(path.stem, {})[part] = path
    return dict(sorted(discovered.items(), key=lambda item: machine_sort_key(item[0] + ".txt")))


def load_matrix(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features in {path}, found shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"NaN or infinite value found in {path}")
    return values


def load_labels(path: Path) -> np.ndarray:
    labels = np.loadtxt(path, dtype=np.int8).reshape(-1)
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Non-binary labels in {path}: {sorted(unique)}")
    return labels


def preprocess(raw_dir: Path, output_dir: Path) -> dict:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    files = discover_files(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    machines_dir = output_dir / "machines"
    machines_dir.mkdir(parents=True, exist_ok=True)

    all_train: list[np.ndarray] = []
    all_test: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    manifest_machines: dict[str, dict] = {}
    train_offset = 0
    test_offset = 0

    for index, (machine_id, machine_files) in enumerate(files.items(), start=1):
        print(f"[{index:02d}/{EXPECTED_MACHINES}] validating {machine_id}")
        train = load_matrix(machine_files["train"])
        test = load_matrix(machine_files["test"])
        labels = load_labels(machine_files["test_label"])
        if len(test) != len(labels):
            raise ValueError(
                f"Test/label length mismatch for {machine_id}: {len(test)} != {len(labels)}"
            )
        intervals = parse_interpretation(machine_files["interpretation_label"])
        if any(item["end"] >= len(test) for item in intervals):
            raise ValueError(f"Interpretation interval exceeds test length for {machine_id}")

        np.savez_compressed(
            machines_dir / f"{machine_id}.npz",
            train=train,
            test=test,
            label=labels,
        )
        manifest_machines[machine_id] = {
            "train_length": int(len(train)),
            "test_length": int(len(test)),
            "train_offset": int(train_offset),
            "test_offset": int(test_offset),
            "anomaly_points": int(labels.sum()),
            "interpretation": intervals,
        }
        train_offset += len(train)
        test_offset += len(test)
        all_train.append(train)
        all_test.append(test)
        all_labels.append(labels)

    train_values = np.concatenate(all_train, axis=0).astype(np.float32, copy=False)
    test_values = np.concatenate(all_test, axis=0).astype(np.float32, copy=False)
    test_labels = np.concatenate(all_labels, axis=0).astype(np.int8, copy=False)
    np.save(output_dir / "SMD_train.npy", train_values)
    np.save(output_dir / "SMD_test.npy", test_values)
    np.save(output_dir / "SMD_test_label.npy", test_labels)

    mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    np.savez(output_dir / "scaler.npz", mean=mean, scale=scale)

    manifest = {
        "schema_version": 1,
        "source": str(raw_dir),
        "machine_count": EXPECTED_MACHINES,
        "feature_count": EXPECTED_FEATURES,
        "sampling_interval_minutes": 1,
        "train_length": int(len(train_values)),
        "test_length": int(len(test_values)),
        "machines": manifest_machines,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    license_path = raw_dir / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, output_dir / "LICENSE")
    print(
        f"Prepared {EXPECTED_MACHINES} machines: train={train_values.shape}, "
        f"test={test_values.shape}, labels={test_labels.shape}"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and preprocess local SMD files")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/SMD"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
