# Monitoring

Everything here is live in the service. `GET /metrics` is Prometheus
exposition; `GET /v1/insights` and `GET /v1/monitoring/drift` are the JSON
views the dashboard below would be built from.

## The one rule: every series carries a version label

With an A/B split running, an aggregate number is a weighted average of two
different models. It moves when the *traffic weights* move, not when either
model changes, which makes it actively misleading during a ramp. So every
counter, histogram and gauge in `src/monitoring/metrics.py` has a `version`
label, and the insight endpoint returns `by_version` alongside `overall`.

## Metrics exposed

| Metric | Type | Labels | What it is for |
| --- | --- | --- | --- |
| `triage_requests_total` | counter | version, category, queue, priority | Volume, and the shape of what each model is routing |
| `triage_records_total` | counter | version | Records classified, independent of batching |
| `triage_low_confidence_total` | counter | version | How often each model falls below the auto-route threshold |
| `triage_human_review_total` | counter | version, reason | Review load, split by `low_confidence` vs `ambiguous_queue_split` |
| `triage_escalations_total` | counter | version | Records sent to the security queue |
| `triage_errors_total` | counter | version, kind | Serving failures |
| `triage_sentiment_total` | counter | version, label | Sentiment mix (uncalibrated — see LIMITATIONS.md) |
| `triage_latency_seconds` | histogram | version, stage | Latency, split into `model` and `post_model` |
| `triage_confidence` | histogram | version | Confidence distribution, the input to drift |
| `triage_ab_traffic_weight` | gauge | version, experiment | Configured split, so a ramp is visible on the same graph as its effect |
| `triage_drift_psi` | gauge | version, feature | PSI against the reference distributions |
| `triage_model_loaded` | gauge | version, family | Which versions are servable |
| `triage_monitoring_window_records` | gauge | — | Occupancy of the in-memory window |

## Dashboard

Four rows. The layout follows the order you actually ask questions in during an
incident: is it up, is it behaving, is the traffic still what we trained on, is
the experiment sound.

**Row 1 — service health.** Request rate by version (`rate(triage_records_total[5m])`),
error rate by version and kind, latency p50/p95/p99 from
`triage_latency_seconds` split by stage. The `model` and `post_model` split
matters: extraction and routing cost the same regardless of which model served
the record, so a latency regression that appears in `post_model` is not the
model's fault.

**Row 2 — decision quality proxies.** These are the numbers that move when a
model degrades, and none of them needs labels:

- auto-route rate by version — `1 - rate(triage_low_confidence_total) / rate(triage_records_total)`
- human review rate by version and reason
- escalation rate by version
- confidence distribution as a heatmap from `triage_confidence`

A drop in auto-route rate with flat volume means the model has become less sure
of traffic it used to handle. That is the earliest signal available without
ground truth.

**Row 3 — traffic shape.** `triage_drift_psi` per feature and version, with
threshold lines at 0.10 and 0.25. Queue mix as a stacked area from
`triage_requests_total` by `queue`. A category mix that shifts without a
corresponding PSI move on text length usually means the model changed its mind,
not that customers did.

**Row 4 — experiment.** `triage_ab_traffic_weight` per version overlaid on the
realised share of `triage_records_total`. These two lines diverging means the
splitter is not doing what the config says, which invalidates the experiment
and is worth an alert on its own.

## Alerts worth setting

| Condition | Why |
| --- | --- |
| `triage_drift_psi > 0.25` for 30 min on any feature | Traffic no longer resembles training data |
| Auto-route rate drops >10 points week on week for one version | Model degrading, or traffic changing under it |
| Realised traffic share differs from `triage_ab_traffic_weight` by >5 points | The split is broken; experiment results are not trustworthy |
| `triage_errors_total` rate > 0 sustained | Serving failure |
| `triage_model_loaded == 0` for a weighted version | A version carrying traffic failed to load |

None of these are configured in this repository — the gauges are exposed, but
nothing pages anyone. See [LIMITATIONS.md](LIMITATIONS.md).

## Drift

Two reference families, both built from artefacts rather than hard-coded:

- **Input drift** — text character and word length, and predicted category mix,
  against the training split (`artifacts/data/processed/drift_reference.json`,
  written by the ETL). Shared across versions, because they trained on the same
  data.
- **Output drift** — confidence distribution, against each version's *own*
  confidences on the validation split
  (`artifacts/models/<version>/drift_reference.json`, written by
  `python -m src.monitoring.build_reference`). Per version: a transformer and a
  TF-IDF model have genuinely different confidence shapes, and comparing either
  against the other's reference would flag drift that is not there.

PSI bands are the conventional ones: below 0.10 stable, 0.10–0.25 moderate,
above 0.25 significant. Below 200 records in the window the reading is returned
with a note saying it is noise, rather than being suppressed — a hidden metric
is worse than a caveated one.

`experiments/run_ab_experiment.py` doubles as a negative control for the
detector: it replays data drawn from the same distribution as training, so the
PSI values it reports should be near zero. If they are not, the detector is
wrong, not the traffic.

## What the insight endpoint adds

`GET /v1/insights` is the operational view rather than the SRE view: volume and
sentiment by category and queue, top intents, review load, and — when records
were replayed from a labelled split — accuracy. Live traffic has no labels, so
`accuracy` is `null` in production, and that is deliberate rather than an
oversight.
