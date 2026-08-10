"""Drive a running triage API and print what comes back, readably.

    python scripts/demo_client.py [--base-url http://localhost:8000]

Used by ``scripts/run_demo.sh``. Talks to the service over HTTP with nothing
but the standard library, so the demo has no dependency the service itself does
not already have.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

SAMPLES = [
    ("My card was stolen this morning and I need it blocked urgently!", "cust-0001"),
    ("What exchange rate do you use for purchases abroad?", "cust-0002"),
    ("I have been charged twice for the same transaction and nobody has replied "
     "in two weeks. This is unacceptable.", "cust-0003"),
    ("asdf qwerty zzz", "cust-0004"),
    ("My new card still has not arrived after 3 weeks", "cust-0005"),
    ("Thanks, the refund came through and everything is sorted now", "cust-0006"),
]


def call(url: str, payload: Optional[Dict[str, Any]] = None, method: str = "GET") -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def call_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def wait_for_ready(base: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            call(f"{base}/readyz")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
            time.sleep(1)
    return False


def show_models(base: str) -> Dict[str, Any]:
    banner("Registered model versions and their share of traffic")
    body = call(f"{base}/v1/models")
    print(f"  experiment: {body['experiment_id']}")
    print(f"  control:    {body['control_version']}")
    print()
    print(f"  {'version':34s} {'family':12s} {'weight':>7s} {'test macro-F1':>14s}")
    print(f"  {'-' * 34} {'-' * 12} {'-' * 7} {'-' * 14}")
    for model in body["models"]:
        f1 = model["test_intent_macro_f1"]
        print(f"  {model['version']:34s} {str(model['family']):12s} "
              f"{model['traffic_weight']:7.2f} "
              f"{(f'{f1:.4f}' if f1 is not None else 'n/a'):>14s}")
    return body


def show_triage(base: str, text: str, client_id: str) -> None:
    result = call(f"{base}/v1/triage", {"text": text, "client_id": client_id}, "POST")
    classification = result["classification"]
    sentiment = result["sentiment"]
    routing = result["routing"]

    alternatives = ", ".join(
        f"{entry['intent']} {entry['score']:.3f}" for entry in classification["top_k"][1:]
    )

    print()
    print(f'  "{text}"')
    print(f"    served by     {result['model_version']}  ({result['ab']['reason']})")
    print(f"    intent        {classification['intent']}  "
          f"conf {classification['confidence']:.3f}  margin {classification['margin']:.3f}")
    print(f"    runners-up    {alternatives or 'none'}")
    print(f"    category      {classification['category']}")
    print(f"    speech act    {classification['speech_act']}   "
          f"urgency {classification['urgency']:.2f} {classification['urgency_markers'] or ''}")
    print(f"    sentiment     {sentiment['label']} ({sentiment['compound']:+.3f}) "
          f"[uncalibrated: is_calibrated={sentiment['is_calibrated']}]")
    print(f"    ROUTED TO     {routing['queue']}  {routing['priority']}  "
          f"SLA {routing['sla_minutes']} min")
    print(f"    human review  {routing['needs_human_review']}   "
          f"auto-routed {routing['auto_routed']}   escalated {routing['escalate']}")
    for reason in routing["reasons"]:
        print(f"                  - {reason}")
    print(f"    latency       {result['latency_ms']:.1f} ms")


def show_insights(base: str) -> None:
    banner("Aggregated insight, separated by model version")
    body = call(f"{base}/v1/insights")
    store = body["store"]
    print(f"  window holds {store['held']} of {store['capacity']} records "
          f"(durable: {store['durable']}, scope: {store['scope']})")

    for version, summary in body["by_version"].items():
        print()
        print(f"  {version}")
        print(f"    records            {summary['n_records']}")
        print(f"    mean confidence    {summary['confidence']['mean']}")
        print(f"    auto-route rate    {summary['auto_route_rate']:.2f}")
        print(f"    escalation rate    {summary['escalation_rate']:.2f}")
        print(f"    human review rate  {summary['confidence']['below_threshold_share']:.2f}")
        print(f"    latency p50/p95    {summary['latency_ms']['p50']} / "
              f"{summary['latency_ms']['p95']} ms")
        queues = {name: stats["count"] for name, stats in summary["by_queue"].items()}
        print(f"    queues             {queues}")
        print(f"    sentiment mix      {summary['sentiment']['mix']}")


def show_ab_shift(base: str, models: Dict[str, Any]) -> None:
    banner("Shifting the traffic split at runtime, without a redeploy")
    control = models["control_version"]
    other = next(
        (m["version"] for m in models["models"]
         if m["version"] != control and m["traffic_weight"] > 0),
        None,
    )
    if other is None:
        print("  only one version carries traffic; nothing to shift")
        return

    print(f"  ramping {other} down to 10% of traffic")
    response = call(
        f"{base}/v1/ab/weights",
        {"weights": {control: 0.9, other: 0.1}},
        "PUT",
    )
    print(f"  new weights: {response['config']['weights']}")
    print(f"  written to config/ab_config.json? {response['persisted']} "
          "(deliberately not: an incident ramp should not become the default)")


def show_drift(base: str) -> None:
    banner("Drift of served traffic against the training reference")
    body = call(f"{base}/v1/monitoring/drift")
    print(f"  bands: stable < {body['thresholds']['stable_below']}, "
          f"significant > {body['thresholds']['significant_above']}")
    any_reading = False
    for version, readings in body["by_version"].items():
        for reading in readings:
            any_reading = True
            note = f"   ({reading['note']})" if reading["note"] else ""
            print(f"  {version:34s} {reading['feature']:24s} "
                  f"PSI {reading['psi']:7.4f}  {reading['band']}{note}")
    if not any_reading:
        print("  no readings yet — run `python -m src.monitoring.build_reference` "
              "and send some traffic")


def show_metrics(base: str) -> None:
    banner("Prometheus metrics (note the version label on every series)")
    interesting = ("triage_records_total", "triage_ab_traffic_weight",
                   "triage_escalations_total", "triage_low_confidence_total")
    shown = 0
    for line in call_text(f"{base}/metrics").splitlines():
        if line.startswith(interesting) and shown < 16:
            print(f"  {line}")
            shown += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--wait", type=int, default=120)
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")

    if not wait_for_ready(base, args.wait):
        print(f"service at {base} never became ready", file=sys.stderr)
        return 1

    models = show_models(base)

    banner("Records through the pipeline: classify, extract, route")
    for text, client_id in SAMPLES:
        show_triage(base, text, client_id)

    # Enough traffic for the aggregations below to say something, and enough
    # distinct client ids for both A/B arms to be represented.
    bulk = [
        {"text": SAMPLES[i % len(SAMPLES)][0], "client_id": f"bulk-{i:04d}"}
        for i in range(60)
    ]
    call(f"{base}/v1/triage/batch", {"items": bulk}, "POST")

    show_insights(base)
    show_metrics(base)
    show_drift(base)
    show_ab_shift(base, models)

    print()
    print(f"Interactive API docs: {base}/docs")
    print("Full reports:         experiments/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
