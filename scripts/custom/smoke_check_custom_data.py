import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

UNITS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(UNITS_ROOT))

from data_provider.data_factory import data_provider


def shape_of(obj):
    if hasattr(obj, "shape"):
        return tuple(obj.shape)
    return type(obj).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "data_provider/custom_timemmd_forecast.yaml",
            "data_provider/custom_mindts_anomaly.yaml",
            "data_provider/custom_all_prompt.yaml",
        ],
    )
    args = parser.parse_args()

    provider_args = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        freq="d",
        target="OT",
        subsample_pct=None,
        fix_seed=2021,
    )

    for config_path in args.configs:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"CONFIG {config_path}")
        for name, task_config in config["task_dataset"].items():
            flag = "test" if task_config["task_name"] == "anomaly_detection" else "train"
            dataset, loader = data_provider(provider_args, task_config, flag=flag, ddp=False)
            first = next(iter(loader))
            shapes = [shape_of(item) for item in first]
            print(f"  {name}: len={len(dataset)} batch_shapes={shapes}")


if __name__ == "__main__":
    main()
