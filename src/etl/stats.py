"""Class-balance and text statistics, written next to the processed splits.

These feed three things: the imbalance-handling section of the README, the
drift reference distribution used by `src/monitoring`, and the sanity tests.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.taxonomy import Taxonomy


def _distribution(counts: pd.Series) -> Dict[str, object]:
    values = counts.to_numpy(dtype=float)
    total = float(values.sum())
    proportions = values / total if total else values

    # Normalised Shannon entropy: 1.0 is perfectly balanced, 0.0 is one class.
    nonzero = proportions[proportions > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = math.log(len(values)) if len(values) > 1 else 1.0

    return {
        "n_classes": int(len(values)),
        "n_rows": int(total),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 2),
        "median": float(np.median(values)),
        "imbalance_ratio_max_over_min": round(float(values.max() / values.min()), 2),
        "normalised_entropy": round(entropy / max_entropy, 4),
        "counts": {str(k): int(v) for k, v in counts.sort_index().items()},
    }


def _text_stats(texts: pd.Series) -> Dict[str, float]:
    lengths = texts.str.len().to_numpy(dtype=float)
    words = texts.str.split().str.len().to_numpy(dtype=float)
    return {
        "chars_mean": round(float(lengths.mean()), 2),
        "chars_p50": float(np.percentile(lengths, 50)),
        "chars_p95": float(np.percentile(lengths, 95)),
        "chars_max": float(lengths.max()),
        "words_mean": round(float(words.mean()), 2),
        "words_p50": float(np.percentile(words, 50)),
        "words_p95": float(np.percentile(words, 95)),
        "words_max": float(words.max()),
    }


def split_stats(frame: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(frame)),
        "intent": _distribution(frame["intent"].value_counts()),
        "category": _distribution(frame["category"].value_counts()),
        "text": _text_stats(frame["text"]),
    }


def build_class_stats(
    splits: Dict[str, pd.DataFrame], taxonomy: Taxonomy
) -> Dict[str, object]:
    return {
        "taxonomy_schema_version": taxonomy.schema_version,
        "n_intents": len(taxonomy.intents),
        "n_categories": len(taxonomy.categories),
        "intents_per_category": {
            key: len(taxonomy.intents_in_category(key))
            for key in taxonomy.category_keys
        },
        "splits": {name: split_stats(frame) for name, frame in splits.items()},
    }


# Bin edges for the drift reference. Fixed rather than quantile-derived so that
# the live histogram and the reference histogram are always directly comparable.
CHAR_BINS = [0, 20, 30, 40, 50, 60, 80, 100, 130, 180, float("inf")]
WORD_BINS = [0, 4, 6, 8, 10, 13, 16, 20, 26, 35, float("inf")]
CONFIDENCE_BINS = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]


def histogram(values: Sequence[float], edges: Sequence[float]) -> List[float]:
    """Proportion of ``values`` falling in each ``edges`` bin. Sums to 1.0."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return [0.0] * (len(edges) - 1)
    counts, _ = np.histogram(array, bins=list(edges))
    total = counts.sum()
    if total == 0:
        return [0.0] * (len(edges) - 1)
    return [round(float(c / total), 6) for c in counts]


def length_reference(texts: pd.Series) -> Dict[str, object]:
    """Reference text-length distribution the drift monitor compares against."""
    return {
        "n_rows": int(len(texts)),
        "chars": {
            "bin_edges": CHAR_BINS,
            "proportions": histogram(texts.str.len().tolist(), CHAR_BINS),
        },
        "words": {
            "bin_edges": WORD_BINS,
            "proportions": histogram(texts.str.split().str.len().tolist(), WORD_BINS),
        },
    }


def class_weights(labels: pd.Series, classes: List[str]) -> np.ndarray:
    """Balanced class weights: n_samples / (n_classes * count(class)).

    Same formula scikit-learn uses for ``class_weight="balanced"``. Returned in
    ``classes`` order so it can be handed straight to a torch loss function.
    """
    counts = labels.value_counts()
    n_samples = float(len(labels))
    n_classes = float(len(classes))
    return np.array(
        [n_samples / (n_classes * max(int(counts.get(c, 0)), 1)) for c in classes],
        dtype=np.float32,
    )
