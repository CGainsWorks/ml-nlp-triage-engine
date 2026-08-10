"""Fetch the raw BANKING77 CSVs and record what was fetched.

The dataset is CC-BY-4.0 but is *not* redistributed by this repository; it is
pulled from upstream at build time. Every download is hashed so a training run
can be tied to exact bytes. See data/DATASET_CARD.md.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Dict

from src.config import BANKING77_BASE_URL, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

SPLITS = ("train", "test")
_TIMEOUT_SECONDS = 60


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_split(split: str, dest_dir: Path = RAW_DATA_DIR, force: bool = False) -> Path:
    """Download one published split, skipping the fetch if it is already there."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{split}.csv"

    if target.exists() and not force:
        LOGGER.info("raw/%s.csv already present, skipping download", split)
        return target

    url = f"{BANKING77_BASE_URL}/{split}.csv"
    LOGGER.info("downloading %s -> %s", url, target)
    tmp = target.with_suffix(".csv.part")
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        with open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp.replace(target)
    return target


def download_all(dest_dir: Path = RAW_DATA_DIR, force: bool = False) -> Dict[str, Path]:
    return {split: download_split(split, dest_dir, force) for split in SPLITS}


def manifest(paths: Dict[str, Path]) -> Dict[str, Dict[str, object]]:
    """Provenance record: URL, local path, size and SHA-256 per file."""
    return {
        split: {
            "url": f"{BANKING77_BASE_URL}/{split}.csv",
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }
        for split, path in paths.items()
    }
