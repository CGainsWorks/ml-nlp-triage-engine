"""Deduplication, leakage control and the train/validation split.

The published BANKING77 test split is used untouched as the held-out set. The
published train split is divided into our train and validation sets, stratified
on the fine intent so every one of the 77 classes appears in both.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, VAL_FRACTION

LOGGER = logging.getLogger(__name__)


def deduplicate(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop exact duplicates and report label-conflicting ones.

    Two kinds of duplicate matter and they are not the same problem:

    * same cleaned text *and* same label -> harmless redundancy, dropped, since
      keeping it just reweights that example.
    * same cleaned text, *different* label -> an irreducible label conflict. We
      keep the first occurrence and count them, because they put a hard ceiling
      on achievable accuracy that no model can cross.
    """
    before = len(frame)
    frame = frame.drop_duplicates(subset=["text", "intent"], keep="first")
    after_exact = len(frame)

    conflict_mask = frame.duplicated(subset=["text"], keep=False)
    conflicting_texts = int(frame.loc[conflict_mask, "text"].nunique())
    conflicting_rows = int(conflict_mask.sum())

    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    stats = {
        "rows_in": before,
        "exact_duplicates_dropped": before - after_exact,
        "label_conflicting_texts": conflicting_texts,
        "label_conflicting_rows_dropped": conflicting_rows - conflicting_texts,
        "rows_out": len(frame),
    }
    return frame, stats


def remove_test_leakage(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop training rows whose cleaned text also appears in the test split.

    Cleaning masks numbers, so two records that differed only by an amount
    collapse to the same string. Any overlap that creates is removed from
    *train*, never from test, so the held-out set stays exactly as published.
    """
    test_texts = set(test["text"])
    overlap = train["text"].isin(test_texts)
    stats = {
        "train_rows_overlapping_test": int(overlap.sum()),
        "distinct_overlapping_texts": int(train.loc[overlap, "text"].nunique()),
    }
    if stats["train_rows_overlapping_test"]:
        LOGGER.warning(
            "dropping %d training rows that collide with test after cleaning",
            stats["train_rows_overlapping_test"],
        )
    return train.loc[~overlap].reset_index(drop=True), stats


def train_val_split(
    frame: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the fine intent."""
    counts = frame["intent"].value_counts()
    too_small = counts[counts < 2]
    if len(too_small):
        raise ValueError(
            "cannot stratify: these intents have fewer than 2 examples: "
            f"{sorted(too_small.index)}"
        )

    train, val = train_test_split(
        frame,
        test_size=val_fraction,
        random_state=seed,
        stratify=frame["intent"],
        shuffle=True,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)
