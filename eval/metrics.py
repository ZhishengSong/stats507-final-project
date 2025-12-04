"""Helpers for computing and exporting classification metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(
    labels: Sequence[int],
    probs: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Accuracy, AUROC, and macro F1."""

    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    preds = (probs_arr >= threshold).astype(int)

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(labels_arr, preds))

    try:
        metrics["auroc"] = float(roc_auc_score(labels_arr, probs_arr))
    except ValueError:
        metrics["auroc"] = float("nan")

    metrics["macro_f1"] = float(f1_score(labels_arr, preds, average="macro"))
    return metrics


def save_predictions_to_csv(
    labels: Sequence[int],
    probs: Sequence[float],
    output_path: str,
) -> None:
    """Write predictions, labels, and probabilities to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = [int(prob >= 0.5) for prob in probs]

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "label", "prob", "pred"])
        for idx, (label, prob, pred) in enumerate(zip(labels, probs, preds)):
            writer.writerow([idx, label, prob, pred])

