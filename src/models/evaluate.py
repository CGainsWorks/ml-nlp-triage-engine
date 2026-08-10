"""Evaluation shared by every model, so the comparison is apples to apples.

Reported at both label levels:

* **intent** (77 classes) — what the model is trained on.
* **category** (9 classes) — the intent prediction mapped through the taxonomy.
  This is what actually determines routing, so it is the number that matters
  operationally even though nothing is trained on it directly.

Headline metric is **macro-F1**, not accuracy. On the intent level the test set
is balanced so the two nearly coincide; on the category level it is not, and
accuracy would quietly over-weight the big queues.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from src.taxonomy import Taxonomy


def _level_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]
) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    # Kappa is undefined when only one class is present in both vectors. Emit
    # null rather than NaN, which is not valid JSON and would break every
    # downstream report that reads these files.
    kappa = float(cohen_kappa_score(y_true, y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "cohen_kappa": round(kappa, 4) if math.isfinite(kappa) else None,
        "n_classes": len(labels),
        "per_class": per_class,
    }


def top_k_accuracy(
    probabilities: np.ndarray, y_true: Sequence[str], labels: List[str], k: int
) -> float:
    index = {label: i for i, label in enumerate(labels)}
    truth = np.array([index[y] for y in y_true])
    k = min(k, probabilities.shape[1])
    top = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    return round(float(np.mean([t in row for t, row in zip(truth, top)])), 4)


def confidence_metrics(
    probabilities: np.ndarray, correct: np.ndarray, thresholds: Sequence[float]
) -> Dict[str, object]:
    """How useful the confidence score is for deciding what to auto-route.

    ``coverage`` is the share of records at or above the threshold;
    ``selective_accuracy`` is accuracy on just those. A confidence score that is
    worth anything buys accuracy faster than it loses coverage.
    """
    confidence = probabilities.max(axis=1)
    rows = []
    for threshold in thresholds:
        mask = confidence >= threshold
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "coverage": round(float(mask.mean()), 4),
                "selective_accuracy": (
                    round(float(correct[mask].mean()), 4) if mask.any() else None
                ),
                "auto_routed_errors": int((~correct & mask).sum()),
            }
        )

    # Expected calibration error, 10 equal-width bins.
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())

    return {
        "mean_confidence": round(float(confidence.mean()), 4),
        "expected_calibration_error": round(float(ece), 4),
        "sweep": rows,
    }


def evaluate(
    y_true_intent: Sequence[str],
    y_pred_intent: Sequence[str],
    probabilities: np.ndarray,
    labels: List[str],
    taxonomy: Taxonomy,
    split: str,
) -> Dict[str, object]:
    y_true_intent = list(y_true_intent)
    y_pred_intent = list(y_pred_intent)
    correct = np.array([t == p for t, p in zip(y_true_intent, y_pred_intent)])

    return {
        "split": split,
        "n_rows": len(y_true_intent),
        "intent": _level_metrics(y_true_intent, y_pred_intent, labels),
        "category": _level_metrics(
            taxonomy.map_intents(y_true_intent),
            taxonomy.map_intents(y_pred_intent),
            taxonomy.category_keys,
        ),
        "top_k_accuracy": {
            f"top_{k}": top_k_accuracy(probabilities, y_true_intent, labels, k)
            for k in (1, 3, 5)
        },
        "confidence": confidence_metrics(
            probabilities, correct, thresholds=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    }


def text_report(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4)
