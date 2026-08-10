"""Head-to-head comparison of every trained model on the held-out test split.

    python -m experiments.compare_baseline_transformer

Writes to ``experiments/results/``:

    model_comparison.md          the table quoted in the README
    model_comparison.json        the same numbers, machine-readable
    per_class_comparison.csv     all 77 intents, both models, signed delta

The per-class file exists because an aggregate macro-F1 hides the interesting
part. A transformer that wins overall by two points can still be *worse* on a
dozen classes, and if those classes are the ones that route to the fraud queue
then the aggregate has told you nothing useful. The Markdown report names the
classes the baseline wins explicitly rather than leaving them in a CSV nobody
opens.

Significance is McNemar's test on paired predictions, which is the right test
here: both models are evaluated on the same records, so their errors are not
independent and a two-sample proportion test would overstate confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2

from src.config import RESULTS_DIR, ensure_dirs
from src.logging_setup import configure
from src.models.data import load_split
from src.models.evaluate import evaluate
from src.models.registry import ModelRegistry
from src.taxonomy import load_taxonomy

LOGGER = logging.getLogger("compare")


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, object]:
    """McNemar's test with continuity correction on paired correctness."""
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    if n == 0:
        return {"b_only_a_correct": b, "c_only_b_correct": c,
                "statistic": None, "p_value": None,
                "note": "models never disagree on this split"}

    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = float(chi2.sf(statistic, df=1))
    return {
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "n_discordant": n,
        "statistic": round(float(statistic), 4),
        "p_value": float(f"{p_value:.6g}"),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def bootstrap_ci(
    correct: np.ndarray, n_resamples: int = 2000, seed: int = 42
) -> Tuple[float, float]:
    """Percentile bootstrap interval for accuracy."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    draws = rng.integers(0, n, size=(n_resamples, n))
    means = correct[draws].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4))


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def measure_latency(
    model, texts: List[str], n_single: int = 200, batch_size: int = 64
) -> Dict[str, float]:
    """Single-record and batched inference cost for one model.

    Single-record is the number that matters for an interactive triage API:
    a customer message arrives on its own, and batching is not available. It is
    also where the transformer's cost shows up most starkly, so quoting only
    batched throughput would flatter it.
    """
    sample = texts[:n_single]
    model.predict_proba(sample[:1])  # warm the inference path before timing

    singles = []
    for text in sample:
        start = time.perf_counter()
        model.predict_proba([text])
        singles.append(1000 * (time.perf_counter() - start))

    batch_start = time.perf_counter()
    for i in range(0, len(sample), batch_size):
        model.predict_proba(sample[i:i + batch_size])
    batched_ms = 1000 * (time.perf_counter() - batch_start) / max(len(sample), 1)

    return {
        "single_record_p50_ms": round(float(np.percentile(singles, 50)), 3),
        "single_record_p95_ms": round(float(np.percentile(singles, 95)), 3),
        "batched_per_record_ms": round(batched_ms, 3),
        "n_timed": len(sample),
        "batch_size": batch_size,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args(argv)

    configure()
    ensure_dirs()
    taxonomy = load_taxonomy()
    registry = ModelRegistry()
    if not registry.versions:
        LOGGER.error("no trained models found; run the training steps first")
        return 1

    frame = load_split(args.split)
    texts = frame["text"].tolist()
    truth = frame["intent"].tolist()

    evaluations: Dict[str, Dict[str, object]] = {}
    correctness: Dict[str, np.ndarray] = {}
    families: Dict[str, str] = {}
    latencies: Dict[str, Dict[str, float]] = {}

    for version in registry.versions:
        model = registry.get(version)
        LOGGER.info("scoring %s over %d %s records", version, len(texts), args.split)
        latencies[version] = measure_latency(model, texts)
        LOGGER.info("%s: %.2f ms per record single, %.2f ms batched",
                    version, latencies[version]["single_record_p50_ms"],
                    latencies[version]["batched_per_record_ms"])
        probabilities = model.predict_proba(texts)
        predicted = [model.labels[i] for i in probabilities.argmax(axis=1)]
        evaluations[version] = evaluate(
            truth, predicted, probabilities, model.labels, taxonomy, args.split
        )
        correctness[version] = np.array([t == p for t, p in zip(truth, predicted)])
        families[version] = model.family
        metadata = registry.metadata(version)
        evaluations[version]["train_seconds"] = (
            metadata.get("train_config") or {}
        ).get("train_seconds")
        evaluations[version]["algorithm"] = (
            metadata.get("train_config") or {}
        ).get("algorithm")

    baselines = [v for v in registry.versions if families[v] == "baseline"]
    transformers = [v for v in registry.versions if families[v] == "transformer"]

    best_baseline = max(
        baselines, key=lambda v: evaluations[v]["intent"]["macro_f1"], default=None
    )
    best_transformer = max(
        transformers, key=lambda v: evaluations[v]["intent"]["macro_f1"], default=None
    )

    comparison: Dict[str, object] = {
        "split": args.split,
        "n_rows": len(texts),
        "models": {
            version: {
                "family": families[version],
                "algorithm": result["algorithm"],
                "train_seconds": result["train_seconds"],
                "intent": {
                    k: result["intent"][k]
                    for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")
                },
                "category": {
                    k: result["category"][k]
                    for k in ("accuracy", "macro_f1", "weighted_f1")
                },
                "top_k_accuracy": result["top_k_accuracy"],
                "expected_calibration_error":
                    result["confidence"]["expected_calibration_error"],
                "mean_confidence": result["confidence"]["mean_confidence"],
                "accuracy_95ci": bootstrap_ci(correctness[version]),
                "latency": latencies[version],
            }
            for version, result in evaluations.items()
        },
    }

    per_class_rows: List[Dict[str, object]] = []
    if best_baseline and best_transformer:
        comparison["head_to_head"] = {
            "baseline": best_baseline,
            "transformer": best_transformer,
            "intent_macro_f1_delta": round(
                evaluations[best_transformer]["intent"]["macro_f1"]
                - evaluations[best_baseline]["intent"]["macro_f1"], 4),
            "category_macro_f1_delta": round(
                evaluations[best_transformer]["category"]["macro_f1"]
                - evaluations[best_baseline]["category"]["macro_f1"], 4),
            "mcnemar": mcnemar(correctness[best_baseline], correctness[best_transformer]),
        }

        base_per_class = evaluations[best_baseline]["intent"]["per_class"]
        tr_per_class = evaluations[best_transformer]["intent"]["per_class"]
        for intent in taxonomy.intents:
            base_f1 = base_per_class[intent]["f1"]
            tr_f1 = tr_per_class[intent]["f1"]
            per_class_rows.append({
                "intent": intent,
                "category": taxonomy.intent_to_category[intent],
                "support": base_per_class[intent]["support"],
                "baseline_f1": base_f1,
                "transformer_f1": tr_f1,
                "delta_transformer_minus_baseline": round(tr_f1 - base_f1, 4),
                "winner": ("transformer" if tr_f1 > base_f1
                           else "baseline" if base_f1 > tr_f1 else "tie"),
            })

        wins = sum(1 for r in per_class_rows if r["winner"] == "transformer")
        losses = sum(1 for r in per_class_rows if r["winner"] == "baseline")
        comparison["head_to_head"]["per_class_intent"] = {
            "transformer_wins": wins,
            "baseline_wins": losses,
            "ties": len(per_class_rows) - wins - losses,
            "baseline_wins_by_more_than_0.05": [
                r["intent"] for r in per_class_rows
                if r["delta_transformer_minus_baseline"] <= -0.05
            ],
        }

    # -- write artefacts -------------------------------------------------

    (RESULTS_DIR / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    if per_class_rows:
        with open(RESULTS_DIR / "per_class_comparison.csv", "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
            writer.writeheader()
            writer.writerows(
                sorted(per_class_rows, key=lambda r: r["delta_transformer_minus_baseline"])
            )

    lines: List[str] = [
        "<!-- GENERATED by `python -m experiments.compare_baseline_transformer` -->\n",
        "# Baseline vs transformer\n\n",
        f"Held-out **{args.split}** split, {len(texts)} records, 77 intents. "
        "No model saw this split during training or model selection.\n\n",
        "## Headline\n\n",
        "| Model | Family | Intent macro-F1 | Intent acc. | Acc. 95% CI | "
        "Category macro-F1 | Top-3 acc. | ECE | Train (s) | p50 ms/record | "
        "batched ms/record |\n",
        "| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    ordered = sorted(
        evaluations, key=lambda v: -evaluations[v]["intent"]["macro_f1"]
    )
    for version in ordered:
        result = evaluations[version]
        entry = comparison["models"][version]
        low, high = entry["accuracy_95ci"]
        lines.append(
            f"| `{version}` | {families[version]} "
            f"| **{_fmt(result['intent']['macro_f1'])}** "
            f"| {_fmt(result['intent']['accuracy'])} "
            f"| {low:.3f}-{high:.3f} "
            f"| {_fmt(result['category']['macro_f1'])} "
            f"| {_fmt(result['top_k_accuracy']['top_3'])} "
            f"| {_fmt(result['confidence']['expected_calibration_error'])} "
            f"| {result['train_seconds'] or 'n/a'} "
            f"| {latencies[version]['single_record_p50_ms']:.2f} "
            f"| {latencies[version]['batched_per_record_ms']:.2f} |\n"
        )

    if "head_to_head" in comparison:
        h2h = comparison["head_to_head"]
        mc = h2h["mcnemar"]
        lines.append(
            f"\n## {h2h['transformer']} vs {h2h['baseline']}\n\n"
            f"Intent macro-F1 delta: **{h2h['intent_macro_f1_delta']:+.4f}** "
            f"(transformer minus baseline).\n"
            f"Category macro-F1 delta: **{h2h['category_macro_f1_delta']:+.4f}**.\n\n"
        )
        if mc.get("p_value") is not None:
            lines.append(
                f"McNemar on paired predictions: {mc['n_discordant']} records where "
                f"exactly one model was right — {mc['b_only_a_correct']} only the "
                f"baseline, {mc['c_only_b_correct']} only the transformer. "
                f"chi-square = {mc['statistic']}, p = {mc['p_value']}. "
                f"{'Significant' if mc['significant_at_0.05'] else 'Not significant'} "
                "at the 0.05 level.\n\n"
            )

        pc = h2h["per_class_intent"]
        lines.append(
            f"### Per-class, all 77 intents\n\n"
            f"Transformer better on **{pc['transformer_wins']}**, "
            f"baseline better on **{pc['baseline_wins']}**, "
            f"tied on **{pc['ties']}**.\n\n"
        )
        # Always name the classes the baseline wins, however small the margin.
        # An aggregate that says "the transformer won" is not the whole truth
        # while there are intents it made worse, and those intents are exactly
        # what someone deciding whether to deploy it needs to look at.
        baseline_wins = sorted(
            (r for r in per_class_rows if r["winner"] == "baseline"),
            key=lambda r: r["delta_transformer_minus_baseline"],
        )
        if baseline_wins:
            worst = baseline_wins[0]["delta_transformer_minus_baseline"]
            lines.append(
                f"**The baseline is better on {len(baseline_wins)} of the 77 "
                f"intents.** The largest single regression is "
                f"{abs(worst):.4f} F1 on `{baseline_wins[0]['intent']}`. "
                "Every one of them is listed here rather than summarised, "
                "because whether these matter depends on which queue they feed, "
                "not on their average:\n\n"
            )
            lines.append(
                "| Intent | Category | Support | Baseline F1 | Transformer F1 | Delta |\n"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: |\n")
            for r in baseline_wins:
                lines.append(
                    f"| `{r['intent']}` | `{r['category']}` | {r['support']} "
                    f"| {r['baseline_f1']:.4f} | {r['transformer_f1']:.4f} "
                    f"| {r['delta_transformer_minus_baseline']:+.4f} |\n"
                )
            lines.append("\n")

            severe = pc["baseline_wins_by_more_than_0.05"]
            lines.append(
                (f"Of these, {len(severe)} exceed 0.05 F1: "
                 + ", ".join(f"`{i}`" for i in severe) + ".\n\n")
                if severe else
                "None of these regressions exceeds 0.05 F1, so on this split "
                "the transformer's wins are broad and its losses are shallow. "
                "That is the case for promoting it — but it is a different "
                "claim from \"the transformer is better everywhere\", which "
                "is not true.\n\n"
            )
        else:
            lines.append("The baseline does not beat the transformer on any intent.\n\n")

        lines.append(
            "Full per-class numbers, sorted worst-delta first: "
            "[`per_class_comparison.csv`](per_class_comparison.csv).\n"
        )

    (RESULTS_DIR / "model_comparison.md").write_text("".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", RESULTS_DIR / "model_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
