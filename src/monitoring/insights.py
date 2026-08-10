"""Aggregated insight reporting over the rolling window.

This is the "what is my inbox actually full of" view: volume by queue, which
categories are running hot, how much is being held for a human, and where the
negative sentiment is concentrated.

Everything here is descriptive. None of it is an accuracy measurement — live
traffic has no labels. The one place accuracy appears is when records were
replayed from a labelled split (`true_intent` set), which is how
`experiments/run_ab_experiment.py` gets per-version accuracy out of the same
code path the service uses.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from src.monitoring.store import PredictionRecord


def _mean(values: List[float]) -> Optional[float]:
    return round(float(np.mean(values)), 4) if values else None


def _breakdown(records: List[PredictionRecord], key: str) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, List[PredictionRecord]] = {}
    for record in records:
        groups.setdefault(str(getattr(record, key)), []).append(record)

    total = len(records)
    out: Dict[str, Dict[str, object]] = {}
    for name, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sentiments = [r.sentiment_compound for r in group]
        out[name] = {
            "count": len(group),
            "share": round(len(group) / total, 4),
            "mean_confidence": _mean([r.confidence for r in group]),
            "mean_sentiment": _mean(sentiments),
            "negative_share": round(
                sum(1 for r in group if r.sentiment_label == "negative") / len(group), 4
            ),
            "human_review_share": round(
                sum(1 for r in group if r.needs_human_review) / len(group), 4
            ),
            "mean_urgency": _mean([r.urgency for r in group]),
        }
    return out


def accuracy_if_labelled(records: List[PredictionRecord]) -> Optional[Dict[str, object]]:
    """Accuracy over replayed records only. ``None`` for live traffic."""
    labelled = [r for r in records if r.true_intent]
    if not labelled:
        return None
    intent_hits = sum(1 for r in labelled if r.intent == r.true_intent)
    return {
        "n_labelled": len(labelled),
        "intent_accuracy": round(intent_hits / len(labelled), 4),
        "note": "replayed labelled traffic only; live records carry no ground truth",
    }


def summarise(records: List[PredictionRecord]) -> Dict[str, object]:
    if not records:
        return {"n_records": 0, "note": "no traffic in this window"}

    timestamps = [r.timestamp for r in records]
    latencies = [r.latency_ms for r in records]
    intents = Counter(r.intent for r in records)

    return {
        "n_records": len(records),
        "window_seconds": round(max(timestamps) - min(timestamps), 2),
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        },
        "confidence": {
            "mean": _mean([r.confidence for r in records]),
            "p05": round(float(np.percentile([r.confidence for r in records], 5)), 4),
            "below_threshold_share": round(
                sum(1 for r in records if r.needs_human_review) / len(records), 4
            ),
        },
        "auto_route_rate": round(
            sum(1 for r in records if r.auto_routed) / len(records), 4
        ),
        "escalation_rate": round(
            sum(1 for r in records if r.escalate) / len(records), 4
        ),
        "sentiment": {
            "mean_compound": _mean([r.sentiment_compound for r in records]),
            "mix": {
                label: round(count / len(records), 4)
                for label, count in Counter(r.sentiment_label for r in records).items()
            },
            "caveat": "uncalibrated lexicon output, not a validated classifier",
        },
        "speech_act_mix": {
            act: round(count / len(records), 4)
            for act, count in Counter(r.speech_act for r in records).most_common()
        },
        "top_intents": [
            {"intent": intent, "count": count, "share": round(count / len(records), 4)}
            for intent, count in intents.most_common(10)
        ],
        "by_category": _breakdown(records, "category"),
        "by_queue": _breakdown(records, "queue"),
        "by_priority": _breakdown(records, "priority"),
        "accuracy": accuracy_if_labelled(records),
    }


def report(
    records: List[PredictionRecord], window_seconds: Optional[float] = None
) -> Dict[str, object]:
    """Overall plus per-version insight. Per-version is the operative view."""
    by_version: Dict[str, List[PredictionRecord]] = {}
    for record in records:
        by_version.setdefault(record.version, []).append(record)

    return {
        "requested_window_seconds": window_seconds,
        "versions_seen": sorted(by_version),
        "overall": summarise(records),
        "by_version": {
            version: summarise(group) for version, group in sorted(by_version.items())
        },
    }
