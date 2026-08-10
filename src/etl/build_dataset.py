"""Build the processed splits from the published BANKING77 CSVs.

    python -m src.etl.build_dataset

Writes to ``artifacts/data/processed/``:

    train.csv val.csv test.csv    text_raw, text, intent, category
    dataset_manifest.json         URLs, sizes, SHA-256, ETL decisions
    class_stats.json              per-split class balance and text statistics
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict

import pandas as pd

from src import __version__
from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    VAL_FRACTION,
    ensure_dirs,
)
from src.etl.clean import clean_text, is_usable
from src.etl.download import download_all, manifest
from src.etl.split import deduplicate, remove_test_leakage, train_val_split
from src.etl.stats import build_class_stats, length_reference
from src.logging_setup import configure
from src.taxonomy import load_taxonomy

LOGGER = logging.getLogger("etl")


def _load_and_clean(path, taxonomy) -> Dict[str, object]:
    frame = pd.read_csv(path)
    if set(frame.columns) != {"text", "category"}:
        raise ValueError(
            f"{path} has unexpected columns {list(frame.columns)}; "
            "expected exactly ['text', 'category']"
        )

    # Upstream calls the fine intent label "category"; we reserve that word for
    # the coarse triage queue, so rename on the way in.
    frame = frame.rename(columns={"category": "intent"})

    unknown = set(frame["intent"]) - set(taxonomy.intent_to_category)
    if unknown:
        raise ValueError(
            "dataset contains intents missing from data/taxonomy.json: "
            f"{sorted(unknown)}"
        )

    frame["text_raw"] = frame["text"].astype(str)
    frame["text"] = frame["text_raw"].map(clean_text)

    usable = frame["text"].map(is_usable)
    dropped_empty = int((~usable).sum())
    frame = frame.loc[usable].reset_index(drop=True)

    frame["category"] = frame["intent"].map(taxonomy.intent_to_category)
    return {"frame": frame[["text_raw", "text", "intent", "category"]],
            "dropped_empty": dropped_empty}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-download", action="store_true",
                        help="re-fetch the raw CSVs even if present")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--expect-sha256", nargs=2, metavar=("TRAIN", "TEST"),
                        help="fail if the downloaded files do not match these hashes")
    args = parser.parse_args(argv)

    configure()
    ensure_dirs()
    taxonomy = load_taxonomy()

    raw_paths = download_all(RAW_DATA_DIR, force=args.force_download)
    provenance = manifest(raw_paths)
    if args.expect_sha256:
        expected = dict(zip(("train", "test"), args.expect_sha256))
        for split, want in expected.items():
            got = provenance[split]["sha256"]
            if got != want:
                raise SystemExit(
                    f"checksum mismatch for {split}.csv: expected {want}, got {got}"
                )
        LOGGER.info("checksums verified against the values passed on the command line")
    else:
        LOGGER.info(
            "checksums recorded but not verified; pass --expect-sha256 to pin them"
        )

    loaded = {split: _load_and_clean(path, taxonomy) for split, path in raw_paths.items()}
    train_full = loaded["train"]["frame"]
    test = loaded["test"]["frame"]
    LOGGER.info("loaded %d train rows, %d test rows after cleaning",
                len(train_full), len(test))

    train_full, train_dedup = deduplicate(train_full)
    test, test_dedup = deduplicate(test)
    train_full, leakage = remove_test_leakage(train_full, test)

    train, val = train_val_split(train_full, args.val_fraction, args.seed)
    splits = {"train": train, "val": val, "test": test}
    LOGGER.info("split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    for name, frame in splits.items():
        out = PROCESSED_DATA_DIR / f"{name}.csv"
        frame.to_csv(out, index=False)
        LOGGER.info("wrote %s (%d rows)", out, len(frame))

    stats = build_class_stats(splits, taxonomy)
    (PROCESSED_DATA_DIR / "class_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # Reference distribution the live drift monitor compares incoming traffic
    # against. Built from train only: it describes what the models were fitted
    # on, which is exactly what "drift" is drift away from.
    (PROCESSED_DATA_DIR / "drift_reference.json").write_text(
        json.dumps(
            {
                "built_from": "train split",
                "text_length": length_reference(train["text"]),
                "category_mix": {
                    "labels": taxonomy.category_keys,
                    "proportions": [
                        round(
                            float((train["category"] == key).sum() / max(len(train), 1)), 6
                        )
                        for key in taxonomy.category_keys
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    record = {
        "pipeline_version": __version__,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "source": provenance,
        "etl": {
            "cleaning": "src/etl/clean.py: NFKC, control strip, quote folding, "
                        "PII/number masking, whitespace collapse",
            "rows_dropped_empty_after_cleaning": {
                split: loaded[split]["dropped_empty"] for split in loaded
            },
            "deduplication": {"train_full": train_dedup, "test": test_dedup},
            "test_leakage_removed_from_train": leakage,
            "split_strategy": (
                "published test split held out untouched; published train split "
                f"divided {1 - args.val_fraction:.0%}/{args.val_fraction:.0%} "
                "stratified on the fine intent"
            ),
        },
        "rows": {name: len(frame) for name, frame in splits.items()},
    }
    (PROCESSED_DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    LOGGER.info("ETL complete -> %s", PROCESSED_DATA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
