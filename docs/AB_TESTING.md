# A/B test design

## Assignment

```
bucket   = sha256("<experiment_id>:<salt>:<unit_id>")[:8] / 2^64      → [0, 1)
version  = first v where bucket < cumulative_weight(v), versions sorted
```

`unit_id` is the `client_id` on the request, falling back to `request_id`. A
request carrying neither is served by the **control** version and its bucket is
recorded as `-1`: an unidentified caller cannot be assigned stably, so putting
it in the treatment arm would add noise to the experiment for no benefit.

Three properties follow from assignment being a pure hash:

**Sticky.** The same customer always gets the same version for the life of the
experiment. A support thread does not switch models halfway through.

**Stateless.** No assignment table, no coordination between replicas. Every
replica computes the same answer from the same inputs, so the split survives
scaling and restarts.

**Reproducible.** `experiments/run_ab_experiment.py` recomputes the exact
assignments the service would have made, which is why the offline analysis
describes the deployed system rather than a re-implementation of it.

Versions are walked in sorted order when accumulating weights. That is not
cosmetic: it means **growing** an arm only pulls in customers who were
previously in a later arm, so ramping the control from 50% to 75% does not
reshuffle anyone already in control. Pinned by
`tests/test_routing.py::test_ramping_keeps_existing_assignments_stable`.

Changing `salt` re-randomises everyone. That is what you want when starting a
genuinely new experiment, and what you must never do mid-experiment.

## Configuration

`config/ab_config.json`:

```json
{
  "experiment_id": "triage-linear-vs-transformer-2026-08",
  "enabled": true,
  "salt": "v1",
  "unit_field": "client_id",
  "control_version": "baseline-linear-v1",
  "weights": {
    "baseline-linear-v1": 0.5,
    "transformer-distilbert-v1": 0.5,
    "baseline-gbdt-v1": 0.0
  }
}
```

Weights must sum to exactly 1.0. They are not normalised silently, because a
set of weights that does not sum to 1 is a config mistake and quietly fixing it
hides the mistake.

`baseline-gbdt-v1` is registered and servable at weight 0. It lost offline;
carrying live traffic on it would only cost accuracy. Keeping it registered
means promoting it is a weight change rather than a deploy.

`PUT /v1/ab/weights` shifts traffic at runtime and deliberately **does not**
write back to the file. A ramp made during an incident should not silently
become the committed default.

## What the experiment measures, and what it cannot

`python -m experiments.run_ab_experiment` replays the held-out test split
through `TriageService` — the same object the HTTP layer uses — with synthetic
customer ids, then reads the results back out of the monitoring window. It
reports, per arm: intent and category accuracy, mean confidence, auto-route
rate, human review rate, escalation rate and latency percentiles.

Three things to hold in mind when reading it.

**It is unpaired.** The two arms see disjoint records. That is what a real A/B
test looks like, and it is statistically weaker than evaluating both models on
the same records. The paired McNemar test in
[`model_comparison.md`](../experiments/results/model_comparison.md) is the
right answer to "which model is better". This experiment answers "does the
deployment behave".

**Accuracy is only available because this is a replay.** The service accepts an
optional `true_intent` field, stores it, and never receives it from live
traffic. In production the accuracy field in the insight report is `null`.

**There is no business metric.** A real decision would be made on handle time,
reopen rate, escalation accuracy or customer satisfaction. None of those exist
in this dataset. Accuracy is a proxy, and a weaker one than it looks: a model
that is more accurate overall but worse on the intents that route to the fraud
queue could easily be the wrong choice.

## Guardrails checked

- **Split balance.** Chi-square goodness of fit of realised counts against
  configured weights. A *high* p-value is the desired outcome. A skewed split
  invalidates everything downstream, so this is checked before any comparison.
- **Determinism.** The experiment re-assigns a probe unit 100 times and aborts
  if the answer ever differs.
- **Drift as a negative control.** The replay is drawn from the training
  distribution, so PSI should be near zero. If the detector fires here, the
  detector is wrong.

## What is missing

No minimum-detectable-effect calculation, no pre-registered stopping rule, no
sequential-testing correction, and no segment analysis. Reading the p-value
after every batch and stopping when it looks good would inflate the false
positive rate; nothing here stops you doing that, so do not.
