"""Central logging setup."""
from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("matplotlib", "PIL", "urllib3", "alembic", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
