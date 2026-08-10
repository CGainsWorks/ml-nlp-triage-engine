# Limitations

Things this project does not do, does not measure, or measures in a way that
should not be over-read. Nothing here is a to-do list in disguise; these are
the honest boundaries of what has been built.

## The data is not a support inbox

BANKING77 was crowdsourced for intent-detection research. Real triage traffic
differs in ways that matter:

- **Single-label.** Every record has exactly one correct intent. Real messages
  routinely carry two ("my card was declined *and* I was charged a fee"), and
  this system will pick one and be scored as correct or incorrect accordingly.
- **Short.** Median 13 tokens, 95th percentile 37. Performance on long
  multi-paragraph complaints is **unmeasured**. The transformer truncates at 64
  tokens, which affects 0.3% of training records here and would affect far more
  in a real inbox.
- **On-topic and clean.** No signatures, no forwarded threads, no attachments,
  no other languages, no abuse, no empty messages.
- **No time dimension.** No timestamps, so no seasonality, no genuine drift to
  detect, and no way to evaluate the drift monitor against a real shift.

## Sentiment is not evaluated

The dataset has **no sentiment labels**. The scorer is VADER plus a small
banking lexicon. It is:

- not trained on this domain,
- not calibrated,
- never evaluated against ground truth, because there is none.

The API returns `is_calibrated: false` on every response for this reason. The
only evidence offered is a face-validity check
([`sentiment_face_validity.md`](../experiments/results/sentiment_face_validity.md))
showing that categories which are complaints by construction score more
negatively than categories which are questions by construction. That shows the
scorer is not broken. It does not show it is right.

**Sentiment nonetheless influences routing** — strongly negative sentiment can
escalate a record by one priority step. That is a real risk: an uncalibrated
signal is affecting an operational decision. It is bounded (escalation is
capped at two steps, and priority is never *lowered*) but it is not eliminated.
Labelling a few thousand records is the honest fix.

## Speech act and urgency are rules, not models

Same reason: no labels, so no accuracy figure. `src/extraction/intent.py` lists
every pattern explicitly. They were written by reading the data, and they will
not transfer to another domain unchanged.

## The coarse categories are ours

The 9 triage categories are a hand-written mapping over the 77 intents. They
are not part of BANKING77 and carry no external validation. Every category
metric in this repository inherits that. A different mapping would produce
different category numbers from the identical model.

## Routing quality is unmeasured

There are no routing labels, so there is no "routing accuracy". What is tested
is that the rules do what they say — the invariants in
`tests/test_routing.py`. Whether the policy is *good* is a question about the
business, and this repository cannot answer it.

## The A/B test measures the wrong thing, on purpose

`experiments/run_ab_experiment.py` replays a labelled split and compares intent
accuracy between arms. A real A/B test would decide on a business outcome —
handle time, reopen rate, escalation accuracy, customer satisfaction — none of
which exists here. Also:

- The arms see **disjoint records**, so the comparison is unpaired and weaker
  than the paired McNemar test in
  [`model_comparison.md`](../experiments/results/model_comparison.md). For
  "which model is better", read that one.
- Customer ids are synthetic and uniformly distributed, so the split is cleaner
  than any real traffic split would be.
- There is no sequential-testing correction, no minimum-detectable-effect
  calculation, and no pre-registered stopping rule.

## Monitoring is single-process and in-memory

`src/monitoring/store.py` is a bounded ring buffer inside one process. With
more than one replica, each holds its own window, and everything is lost on
restart. The API reports this in every response (`store.durable: false`) rather
than leaving it to be discovered. Production would write the same records to a
queue or an append-only table and aggregate there; the interface is narrow
enough that this is a small change, but it **has not been made**.

## Drift detection catches input drift only

PSI over text length, predicted category mix and confidence. That will catch
traffic starting to look different. It will **not** catch concept drift — the
same inputs coming to mean something else — because detecting that requires
labels arriving after the fact, and there is no feedback loop here.

The predicted-category-mix reading is additionally confounded: it compares
*predicted* categories against the *training label* mix, so model error and
genuine drift both move it. The reading is annotated to say so.

## Calibration

Expected calibration error is reported for every model, and the confidence
threshold is chosen against an explicit cost model
([`confidence_threshold.md`](../experiments/results/confidence_threshold.md)).
No calibration is applied — no temperature scaling, no Platt scaling, no
conformal prediction. The threshold is applied to raw softmax or raw
`predict_proba` output.

## Not built

- No authentication, rate limiting, or request quotas on the API.
- No model rollback mechanism beyond editing the A/B weights.
- No automated retraining, and no scheduled drift alerting — the PSI numbers
  are exposed as Prometheus gauges, but nothing pages anyone.
- No GPU path is exercised. The code will use CUDA if it finds it; that has not
  been tested, and every reported timing is CPU.
- No multilingual support. English only.
