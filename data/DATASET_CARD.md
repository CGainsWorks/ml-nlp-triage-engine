# Dataset card

## BANKING77

| Field | Value |
| --- | --- |
| Name | BANKING77 |
| Upstream source | `https://github.com/PolyAI-LDN/task-specific-datasets` (`banking_data/`) |
| Publisher | PolyAI Limited |
| Paper | Casanueva, Temčinas, Gerz, Henderson, Vulić — *Efficient Intent Detection with Dual Sentence Encoders* (NLP4ConvAI, ACL 2020) |
| **Licence** | **Creative Commons Attribution 4.0 International (CC-BY-4.0)** |
| Licence URL | `https://creativecommons.org/licenses/by/4.0/` |
| Commercial use | Permitted under CC-BY-4.0, with attribution |
| Size | 13,083 rows (10,003 train / 3,080 test as published) |
| Labels | 77 fine-grained banking intents, single-label |
| Language | English |
| Fields | `text` (customer message), `category` (intent label) |

### Attribution

> BANKING77 © PolyAI Limited, licensed under CC-BY-4.0.
> Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, 2020.

### How it is obtained

`python -m src.etl.build_dataset` downloads `train.csv` and `test.csv` from the
upstream repository at build time into `artifacts/data/raw/`. **Neither the raw
data nor any derived split is committed to this repository or attached to a
release.** The SHA-256 of each downloaded file is recorded in
`artifacts/data/processed/dataset_manifest.json` so a run can be shown to have
used the same bytes.

Checksums observed at time of writing (asserted by the ETL when
`--verify-checksums` is passed, warned about otherwise, because the upstream
repository could legitimately update):

```
train.csv  see artifacts/data/processed/dataset_manifest.json after a run
test.csv   see artifacts/data/processed/dataset_manifest.json after a run
```

### What we changed

The published split is used as-is for **test**. The published **train** file is
split 85/15 stratified by intent into our train and validation sets with a fixed
seed (`--seed 42`). We never touch the test set during model selection.

Cleaning is deliberately light — see `src/etl/clean.py`. We normalise Unicode,
collapse whitespace, strip control characters and mask numeric spans that look
like card numbers, sort codes, amounts and dates. We do **not** lowercase, stem,
or remove stop words in the ETL; casing is a useful signal for the transformer
and the TF-IDF vectoriser does its own lowercasing.

### Class balance

The published **test** set is exactly balanced: 40 examples for each of the 77
intents. The published **train** set is not: between 35 and 187 examples per
intent. Exact per-class counts are written to
`artifacts/data/processed/class_stats.json` by the ETL.

This matters for reading the numbers in the README:

- Because test is balanced, macro-F1 and accuracy over the 77 intents are close
  together, and macro-F1 is not inflated by a dominant class.
- The **coarse category** view (9 triage queues, derived — see below) is *not*
  balanced in test, because categories contain different numbers of intents
  (5 to 11). Macro-F1 over categories is therefore the honest headline there,
  not accuracy.

### Derived labels — read this before quoting category numbers

BANKING77 ships **one** label per row: the fine intent. The 9 coarse triage
categories in `data/taxonomy.json` are **our own deterministic mapping** from
those 77 intents, written by hand for this project. They are not part of the
dataset and carry no external validation. Category predictions are produced by
mapping the model's predicted intent through that table, not by a separately
trained category model.

### What the dataset does not contain

- **No sentiment labels.** Any sentiment number this project reports is from an
  unsupervised lexicon, is uncalibrated on this domain, and is not evaluated
  against ground truth. See `docs/LIMITATIONS.md`.
- **No urgency, priority or routing labels.** Routing decisions are rule-based
  on top of the predicted category; there is no held-out routing accuracy to
  report, because there is nothing to compare against.
- **No customer identifiers, timestamps or metadata.** The A/B experiment in
  this repository therefore assigns synthetic request identifiers.

### Known limitations of the data itself

- Single-label. Real triage traffic is frequently multi-intent; a message asking
  about both a fee and a failed transfer has one correct answer here.
- Short utterances (median well under 20 tokens). Performance on long,
  multi-paragraph complaint narratives is unmeasured by this project.
- Written in a single domain (retail/challenger banking) and a single language.
- Collected via crowdsourcing for intent-detection research, so it is cleaner
  and more on-topic than a real support inbox.

## Other data used

None. No second dataset is downloaded, and no synthetic training data is
generated. The VADER sentiment lexicon shipped inside the `vaderSentiment`
package (MIT licence, Hutto & Gilbert, ICWSM 2014) is used at inference time as
a lexicon, not trained on.
