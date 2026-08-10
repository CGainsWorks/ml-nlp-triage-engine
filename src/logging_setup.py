"""One place to configure logging so every entry point looks the same."""

from __future__ import annotations

import logging
import os


def configure(level: str = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("TRIAGE_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # These two are chatty and say nothing useful during training.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
