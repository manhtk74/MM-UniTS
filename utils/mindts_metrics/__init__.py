import math

import numpy as np

from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .vus_metrics import generate_curve


def get_list_anomaly(labels):
    labels = np.asarray(labels, dtype=int)
    end_pos = np.diff(labels, append=0) < 0
    lengths = np.diff(np.cumsum(labels)[end_pos], prepend=0)
    return lengths if len(lengths) > 0 else np.array([1])


def affiliation_f(actual, predicted, score=None):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    result = pr_from_events(events_pred, events_label, (0, len(predicted)))
    precision = result["precision"]
    recall = result["recall"]
    if math.isnan(precision) or math.isnan(recall):
        return 0.0
    denom = precision + recall
    return 0.0 if denom == 0 else 2 * precision * recall / denom


def vus_roc(actual, predicted, score):
    sliding_window = int(np.median(get_list_anomaly(actual)))
    _, _, _, _, _, _, value, _ = generate_curve(actual, score, 2 * sliding_window)
    return value


def vus_pr(actual, predicted, score):
    sliding_window = int(np.median(get_list_anomaly(actual)))
    _, _, _, _, _, _, _, value = generate_curve(actual, score, 2 * sliding_window)
    return value
