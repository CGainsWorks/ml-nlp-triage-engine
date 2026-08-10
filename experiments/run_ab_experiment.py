"""Replay the labelled test split through the live service path as an A/B test.

    python -m experiments.run_ab_experiment

Every record is given a synthetic customer id and pushed through the same
:class:`~src.api.service.TriageService` the HTTP API uses, so the A/B split,
the routing rules and the monitoring window all behave exactly as they do in
production. The results are then read back out of the monitoring layer rather
than computed on the side, which is the point: it tests the monitoring as much
as it tests the models.

**How to read the accuracy comparison.** The two arms see *disjoint* records,
so this is an unpaired comparison and it is deliberately weaker than the
paired McNemar test in `compare_baseline_transformer.py`. If you want to know
which model is better, read that one. What this experiment answers is
different: does the deployed system split traffic the way it was configured to,
does each arm behave sanely once routing is applied, and does per-version
monitoring actually separate the two.

Live traffic has no labels; this replay has them only because it is a replay.
The service records `true_intent` for exactly this purpose and never receives
it in production.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

import numpy as np
from scipy.stats import chisquare, norm

from src.api.service import TriageInput, TriageService
from src.config import RESULTS_DIR, ensure_dirs
from src.logging_setup import configure
from src.models.data import load_split
from src.models.registry import ModelRegistry
from src.monitoring import insights
from src.monitoring.drift import DriftMonitor
from src.monitoring.metrics import TriageMetrics
from src.monitoring.store import PredictionStore
from src.routing.ab import assign, load_ab_config
from src.routing.rules import RoutingPolicy
from src.taxonomy import load_taxonomy

LOGGER = logging.getLogger("ab")


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> Dict[str, object]:
    """Unpaired two-sided z-test on the difference of two proportions."""
    if n1 == 0 or n2 == 0:
        return {"p_value": None, "note": "an arm received no traffic"}
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    denominator = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if denominator == 0:
        return {"p_value": None, "note": "zero variance"}
    z = (p1 - p2) / denominator
    p_value = float(2 * norm.sf(abs(z)))

    # Wald interval on the difference, using unpooled variance.
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "arm_a_rate": round(p1, 4),
        "arm_b_rate": round(p2, 4),
        "difference": round(p2 - p1, 4),
        "difference_95ci": [round(float((p2 - p1) - 1.96 * se), 4),
                            round(float((p2 - p1) + 1.96 * se), 4)],
        "z": round(float(z), 4),
        "p_value": float(f"{p_value:.6g}"),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--unit-prefix", default="cust-")
    args = parser.parse_args(argv)

    configure()
    ensure_dirs()

    registry = ModelRegistry()
    ab_config = load_ab_config()
    missing = set(ab_config.active_versions) - set(registry.versions)
    if missing:
        LOGGER.error("A/B config needs versions that are not trained: %s", sorted(missing))
        return 1
    registry.preload(ab_config.active_versions)

    store = PredictionStore(capacity=100_000)
    service = TriageService(
        registry=registry,
        ab_config=ab_config,
        taxonomy=load_taxonomy(),
        policy=RoutingPolicy.from_file(),
        metrics=TriageMetrics(),
        store=store,
        drift=DriftMonitor.from_artifacts(registry.versions),
    )

    frame = load_split(args.split)
    LOGGER.info("replaying %d %s records through the service", len(frame), args.split)

    # Stickiness check before anything else: the same unit must always land in
    # the same arm, otherwise every number below is meaningless.
    probe = f"{args.unit_prefix}000000"
    first = assign(probe, ab_config).version
    if any(assign(probe, ab_config).version != first for _ in range(100)):
        LOGGER.error("A/B assignment is not deterministic")
        return 1

    items = [
        TriageInput(
            text=row.text_raw if isinstance(row.text_raw, str) else row.text,
            client_id=f"{args.unit_prefix}{i:06d}",
            request_id=f"req-{i:06d}",
            true_intent=row.intent,
        )
        for i, row in enumerate(frame.itertuples(index=False))
    ]

    for start in range(0, len(items), args.batch_size):
        service.triage_many(items[start:start + args.batch_size])
        if (start // args.batch_size) % 5 == 0:
            LOGGER.info("  %d/%d", min(start + args.batch_size, len(items)), len(items))

    records = store.snapshot()
    if not records:
        # Writing a report full of NaN would look like a result. It is not one.
        LOGGER.error(
            "replayed %d records but the monitoring window is empty — the "
            "service is not writing to the store it was given", len(items)
        )
        return 1

    by_version: Dict[str, List] = {}
    for record in records:
        by_version.setdefault(record.version, []).append(record)

    # -- split balance ---------------------------------------------------

    configured = {v: w for v, w in ab_config.weights.items() if w > 0}
    observed_counts = [len(by_version.get(v, [])) for v in configured]
    expected_counts = [w * len(records) for w in configured.values()]
    balance = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    split_report = {
        "n_records": len(records),
        "configured_weights": configured,
        "observed": {
            v: {
                "count": len(by_version.get(v, [])),
                "share": round(len(by_version.get(v, [])) / max(len(records), 1), 4),
            }
            for v in configured
        },
        "chi_square": {
            "statistic": round(float(balance.statistic), 4),
            "p_value": float(f"{float(balance.pvalue):.6g}"),
            "interpretation": (
                "a high p-value means the realised split is consistent with the "
                "configured weights, which is what we want here"
            ),
        },
    }

    # -- per-arm behaviour ------------------------------------------------

    taxonomy = load_taxonomy()
    per_version: Dict[str, Dict[str, object]] = {}
    for version, group in sorted(by_version.items()):
        labelled = [r for r in group if r.true_intent]
        intent_hits = sum(1 for r in labelled if r.intent == r.true_intent)
        category_hits = sum(
            1 for r in labelled
            if r.category == taxonomy.intent_to_category[r.true_intent]
        )
        per_version[version] = {
            "n": len(group),
            "n_labelled": len(labelled),
            "intent_accuracy": round(intent_hits / max(len(labelled), 1), 4),
            "category_accuracy": round(category_hits / max(len(labelled), 1), 4),
            "intent_hits": intent_hits,
            "mean_confidence": round(float(np.mean([r.confidence for r in group])), 4),
            "auto_route_rate": round(
                sum(1 for r in group if r.auto_routed) / len(group), 4),
            "human_review_rate": round(
                sum(1 for r in group if r.needs_human_review) / len(group), 4),
            "escalation_rate": round(
                sum(1 for r in group if r.escalate) / len(group), 4),
            "latency_ms_p50": round(
                float(np.percentile([r.latency_ms for r in group], 50)), 3),
            "latency_ms_p95": round(
                float(np.percentile([r.latency_ms for r in group], 95)), 3),
        }

    arms = sorted(per_version)
    comparisons = {}
    if len(arms) >= 2:
        a, b = ab_config.control_version, next(v for v in arms if v != ab_config.control_version)
        comparisons["intent_accuracy"] = {
            "arm_a": a, "arm_b": b,
            **two_proportion_test(
                per_version[a]["intent_hits"], per_version[a]["n_labelled"],
                per_version[b]["intent_hits"], per_version[b]["n_labelled"],
            ),
        }

    payload = {
        "experiment_id": ab_config.experiment_id,
        "split": args.split,
        "split_balance": split_report,
        "per_version": per_version,
        "comparisons": comparisons,
        "insights": insights.report(records),
        "drift": service.drift_report(),
        "caveats": [
            "arms see disjoint records, so this is an unpaired comparison and is "
            "less powerful than the paired McNemar test in model_comparison.md",
            "accuracy is only computable because this is a replay of a labelled "
            "split; live traffic carries no ground truth",
            "there is no business outcome metric here (handle time, reopen rate, "
            "customer satisfaction), which is what a real A/B test would decide on",
        ],
    }

    # -- report ----------------------------------------------------------

    lines = [
        "<!-- GENERATED by `python -m experiments.run_ab_experiment` -->\n",
        "# A/B test: live-path replay\n\n",
        f"Experiment `{ab_config.experiment_id}` — {len(records)} records from the "
        f"**{args.split}** split replayed through `TriageService`, the same code "
        "path the HTTP API uses.\n\n",
        "## Traffic split\n\n",
        "| Version | Configured | Observed | Records |\n| --- | ---: | ---: | ---: |\n",
    ]
    for version, weight in configured.items():
        observed = split_report["observed"][version]
        lines.append(
            f"| `{version}` | {weight:.2f} | {observed['share']:.4f} "
            f"| {observed['count']} |\n"
        )
    lines.append(
        f"\nChi-square goodness of fit against the configured weights: "
        f"statistic {split_report['chi_square']['statistic']}, "
        f"p = {split_report['chi_square']['p_value']}. A high p-value is the "
        "desired outcome — it means the hash-based splitter is not skewed.\n\n"
    )

    lines.append("## Per-version behaviour\n\n")
    lines.append("| Version | n | Intent acc. | Category acc. | Mean conf. | "
                 "Auto-route | Human review | Escalated | p50 ms | p95 ms |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for version in arms:
        m = per_version[version]
        lines.append(
            f"| `{version}` | {m['n']} | {m['intent_accuracy']:.4f} "
            f"| {m['category_accuracy']:.4f} | {m['mean_confidence']:.4f} "
            f"| {m['auto_route_rate']:.4f} | {m['human_review_rate']:.4f} "
            f"| {m['escalation_rate']:.4f} | {m['latency_ms_p50']:.1f} "
            f"| {m['latency_ms_p95']:.1f} |\n"
        )

    if "intent_accuracy" in comparisons:
        c = comparisons["intent_accuracy"]
        lines.append(
            f"\n### `{c['arm_b']}` against control `{c['arm_a']}`\n\n"
            f"Intent accuracy {c['arm_a_rate']:.4f} vs {c['arm_b_rate']:.4f}, "
            f"difference **{c['difference']:+.4f}** "
            f"(95% CI {c['difference_95ci'][0]:+.4f} to {c['difference_95ci'][1]:+.4f}), "
            f"z = {c['z']}, p = {c['p_value']}. "
            f"{'Significant' if c['significant_at_0.05'] else 'Not significant'} "
            "at the 0.05 level.\n\n"
            "This is the unpaired test. For the question \"which model is "
            "better\", the paired comparison in "
            "[`model_comparison.md`](model_comparison.md) is the one to use.\n\n"
        )

    lines.append("## Caveats\n\n")
    for caveat in payload["caveats"]:
        lines.append(f"- {caveat}\n")

    lines.append(
        "\n## Monitoring output\n\n"
        "Per-version insight aggregation and drift readings for this run are in "
        "[`ab_experiment.json`](ab_experiment.json) under `insights` and `drift`. "
        "Drift is near zero by construction here: the replay is drawn from the "
        "same distribution as training, so this run is a control that shows the "
        "detector does *not* fire on data it should not fire on.\n"
    )

    (RESULTS_DIR / "ab_experiment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "ab_experiment.md").write_text("".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", RESULTS_DIR / "ab_experiment.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
