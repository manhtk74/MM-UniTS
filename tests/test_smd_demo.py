from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import SMDAnomalyDemoLoader, SMDForecastDemoLoader
from demo.alert_engine import build_incidents, status_at
from demo.generate_artifacts import baseline_outputs
from scripts.smd.prepare_smd import parse_interpretation


class InterpretationTests(unittest.TestCase):
    def test_parse_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text("10-12:1,3,38\n20-21:2", encoding="utf-8")
            parsed = parse_interpretation(path)
        self.assertEqual(parsed[0], {"start": 10, "end": 12, "dimensions": [1, 3, 38]})
        self.assertEqual(len(parsed), 2)


class AlertEngineTests(unittest.TestCase):
    def test_warning_critical_and_recovery(self):
        scores = np.zeros(140, dtype=np.float32)
        scores[100:105] = 2.0
        residual = np.zeros((140, 38), dtype=np.float32)
        residual[100:105, 8] = 4.0
        incidents = build_incidents(scores, 1.0, residual)
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["severity"], "CRITICAL")
        self.assertEqual(incidents[0]["start"], 100)
        self.assertEqual(incidents[0]["end"], 104)
        self.assertEqual(incidents[0]["top_kpis"][0], "KPI_09")
        self.assertEqual(status_at(incidents, 102, 2.0, 1.0), "CRITICAL")
        self.assertEqual(status_at(incidents, 108, 0.0, 1.0), "RECOVERED")


class LoaderContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "machines").mkdir()
        machines = {}
        for index, machine_id in enumerate(("machine-1-1", "machine-1-2"), start=1):
            train = np.full((160, 38), index, dtype=np.float32)
            test = np.full((140, 38), index * 10, dtype=np.float32)
            label = np.zeros(140, dtype=np.uint8)
            np.savez_compressed(root / "machines" / f"{machine_id}.npz",
                                train=train, test=test, label=label)
            machines[machine_id] = {
                "train_length": 160,
                "test_length": 140,
                "interpretation": [],
            }
        (root / "manifest.json").write_text(json.dumps({"machines": machines}), encoding="utf-8")
        np.savez(root / "scaler.npz", mean=np.zeros(38, dtype=np.float32),
                 scale=np.ones(38, dtype=np.float32))
        self.root = root

    def tearDown(self):
        self.temp.cleanup()

    def test_anomaly_windows_do_not_cross_machine_boundary(self):
        dataset = SMDAnomalyDemoLoader(self.root, win_size=96, step=50, flag="test")
        self.assertEqual(len(dataset), 2)
        first, first_label = dataset[0]
        second, _ = dataset[1]
        self.assertEqual(first.shape, (96, 38))
        self.assertEqual(first_label.shape, (96,))
        self.assertTrue(np.all(first == 10))
        self.assertTrue(np.all(second == 20))

    def test_forecast_contract_and_boundary(self):
        dataset = SMDForecastDemoLoader(
            self.root, flag="test", size=[10, 5, 3], step=4
        )
        seq_x, seq_y, x_mark, y_mark = dataset[0]
        self.assertEqual(seq_x.shape, (10, 38))
        self.assertEqual(seq_y.shape, (8, 38))
        self.assertEqual(x_mark.shape, (10, 4))
        self.assertEqual(y_mark.shape, (8, 4))
        first_machine_windows = dataset.window_counts[0]
        next_machine, *_ = dataset[first_machine_windows]
        self.assertTrue(np.all(next_machine == 20))


class BaselineTests(unittest.TestCase):
    def test_baseline_shapes(self):
        rng = np.random.default_rng(4)
        train = rng.normal(size=(180, 38)).astype(np.float32)
        test = rng.normal(size=(160, 38)).astype(np.float32)
        outputs = baseline_outputs(train, test, np.ones(38, dtype=np.float32))
        expected, residual, scores, threshold, origins, forecasts = outputs
        self.assertEqual(expected.shape, test.shape)
        self.assertEqual(residual.shape, test.shape)
        self.assertEqual(scores.shape, (len(test),))
        self.assertGreater(threshold, 0)
        self.assertEqual(forecasts.shape, (len(origins), 24, 38))


if __name__ == "__main__":
    unittest.main()
