# ml-nlp-triage-engine

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

A text triage service that classifies incoming records, extracts sentiment and intent and routes accordingly. Scikit-learn baselines establish the floor, transformers provide the lift, and the deployed service A/B tests model versions with per-version monitoring.

---

## What this covers

CV entries this repository is the evidence for:

- NLP Triage and Insight Engine

---

## Requirements

### Hardware

- None beyond a development machine
- GPU optional for training

### Software

- Hugging Face Transformers
- scikit-learn baselines
- Pandas and NumPy ETL
- FastAPI
- Docker
- A/B routing layer
- Monitoring and metrics

### Report and documentation deliverables

- Baseline against transformer comparison
- Class-imbalance handling
- Per-class metrics
- A/B test design and results
- Monitoring dashboard description

---

## Inputs

- Raw text records
- Label taxonomy
- Training and validation splits
- Model version routing weights

## Outputs

- Category prediction
- Sentiment and intent scores
- Confidence
- Routing decision
- Aggregated insight reports
- Per-version performance metrics

---

## Intended structure

```
src/
  etl/                    Pandas/NumPy pipeline, cleaning, splits
  models/
    baseline/             TF-IDF + linear, gradient boosting
    transformer/          fine-tuning and inference
  extraction/             sentiment, intent
  routing/                triage rules, A/B assignment
  api/                    FastAPI service
  monitoring/             per-version metrics, drift
experiments/              baseline vs transformer results
data/                     DATASET_CARD.md, TAXONOMY.md
docker/
docs/
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Label taxonomy defined
- [ ] ETL pipeline with cleaning and splits
- [ ] scikit-learn baseline with reported metrics
- [ ] Transformer fine-tuning and comparison against baseline
- [ ] Class-imbalance handling
- [ ] Sentiment and intent extraction
- [ ] Routing logic
- [ ] FastAPI service with A/B version assignment
- [ ] Per-version monitoring and metrics
- [ ] A/B test analysis
- [ ] Aggregated insight reporting

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: ml-nlp-triage-engine
A text triage service that classifies incoming records, extracts sentiment and intent and routes accordingly. Scikit-learn baselines establish the floor, transformers provide the lift, and the deployed service A/B tests model versions with per-version monitoring.

TARGET STACK
- Hugging Face Transformers
- scikit-learn baselines
- Pandas and NumPy ETL
- FastAPI
- Docker
- A/B routing layer
- Monitoring and metrics

HARDWARE CONTEXT
- None beyond a development machine
- GPU optional for training

THE SYSTEM MUST TAKE THESE INPUTS
- Raw text records
- Label taxonomy
- Training and validation splits
- Model version routing weights

AND PRODUCE THESE OUTPUTS
- Category prediction
- Sentiment and intent scores
- Confidence
- Routing decision
- Aggregated insight reports
- Per-version performance metrics

INTENDED REPOSITORY LAYOUT
src/
  etl/                    Pandas/NumPy pipeline, cleaning, splits
  models/
    baseline/             TF-IDF + linear, gradient boosting
    transformer/          fine-tuning and inference
  extraction/             sentiment, intent
  routing/                triage rules, A/B assignment
  api/                    FastAPI service
  monitoring/             per-version metrics, drift
experiments/              baseline vs transformer results
data/                     DATASET_CARD.md, TAXONOMY.md
docker/
docs/

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
