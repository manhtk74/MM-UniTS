"""Incident aggregation rules shared by artifact generation and the UI."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _finalize_incident(incident, scores, residual, top_k):
    start = int(incident["start"])
    end = int(max(start, incident["end"]))
    segment = residual[start:end + 1]
    contribution = np.mean(np.square(segment), axis=0) if len(segment) else np.zeros(residual.shape[1])
    top_indices = np.argsort(contribution)[::-1][:top_k]
    return {
        "start": start,
        "end": end,
        "duration": end - start + 1,
        "severity": incident["severity"],
        "status": "RECOVERED",
        "max_score": float(np.max(scores[start:end + 1])),
        "top_kpis": [f"KPI_{index + 1:02d}" for index in top_indices],
        "top_contributions": [float(contribution[index]) for index in top_indices],
    }


def build_incidents(
    scores: Iterable[float],
    threshold: float,
    residual: np.ndarray,
    warmup: int = 96,
    warning_points: int = 2,
    critical_points: int = 3,
    critical_multiplier: float = 1.5,
    recovery_points: int = 5,
    merge_gap: int = 10,
    top_k: int = 5,
) -> list[dict]:
    """Aggregate point scores into stable, de-duplicated incidents."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2 or len(residual) != len(scores):
        raise ValueError("residual must have shape [time, features] and match scores")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite number")

    raw_incidents = []
    current = None
    over_run = critical_run = recovery_run = 0
    for index, score in enumerate(scores):
        if index < warmup:
            continue
        if score > threshold:
            recovery_run = 0
            over_run += 1
            critical_run = critical_run + 1 if score > critical_multiplier * threshold else 0
            if current is None and over_run >= warning_points:
                current = {
                    "start": index - over_run + 1,
                    "end": index,
                    "severity": "WARNING",
                }
            if current is not None:
                current["end"] = index
                if critical_run >= critical_points:
                    current["severity"] = "CRITICAL"
        else:
            over_run = 0
            critical_run = 0
            if current is not None:
                recovery_run += 1
                if recovery_run >= recovery_points:
                    current["end"] = index - recovery_run
                    raw_incidents.append(current)
                    current = None
                    recovery_run = 0
            else:
                recovery_run = 0
    if current is not None:
        current["end"] = len(scores) - 1
        raw_incidents.append(current)

    merged = []
    for incident in raw_incidents:
        if merged and incident["start"] - merged[-1]["end"] - 1 <= merge_gap:
            merged[-1]["end"] = incident["end"]
            if incident["severity"] == "CRITICAL":
                merged[-1]["severity"] = "CRITICAL"
        else:
            merged.append(dict(incident))
    return [_finalize_incident(item, scores, residual, top_k) for item in merged]


def status_at(incidents: list[dict], cursor: int, score: float, threshold: float) -> str:
    for incident in incidents:
        if incident["start"] <= cursor <= incident["end"]:
            return incident["severity"]
    if score > threshold:
        return "WARNING"
    if any(incident["end"] < cursor <= incident["end"] + 5 for incident in incidents):
        return "RECOVERED"
    return "NORMAL"
