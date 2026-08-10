"""Load the processed splits produced by ``python -m src.etl.build_dataset``."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.config import PROCESSED_DATA_DIR

SPLITS = ("train", "val", "test")


def load_split(name: str) -> pd.DataFrame:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.etl.build_dataset"
        )
    frame = pd.read_csv(path)
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame


def load_all() -> Dict[str, pd.DataFrame]:
    return {name: load_split(name) for name in SPLITS}


def texts_and_labels(frame: pd.DataFrame) -> "tuple[List[str], List[str]]":
    return frame["text"].tolist(), frame["intent"].tolist()
