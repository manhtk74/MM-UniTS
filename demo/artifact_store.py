"""Read-only access to precomputed demo artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {self.root}")

    def backends(self) -> list[str]:
        return sorted(
            path.name for path in self.root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )

    def backend_manifest(self, backend: str) -> dict:
        path = self.root / backend / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def machines(self, backend: str) -> list[str]:
        return list(self.backend_manifest(backend)["machines"])

    def load(self, backend: str, machine_id: str) -> dict:
        directory = self.root / backend / machine_id
        if not directory.is_dir():
            raise FileNotFoundError(f"No artifact for {backend}/{machine_id}")
        with np.load(directory / "series.npz") as values:
            result = {name: values[name] for name in values.files}
        incident_payload = json.loads((directory / "incidents.json").read_text(encoding="utf-8"))
        result["incidents"] = incident_payload.get("incidents", [])
        result["interpretation"] = incident_payload.get("ground_truth_interpretation", [])
        result["metrics"] = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        return result
