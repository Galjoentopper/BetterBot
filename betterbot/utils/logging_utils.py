"""Reusable logging configuration helpers for BetterBot."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[str | Path] = None) -> None:
    """Configure the root logger with sensible defaults.

    Parameters
    ----------
    level:
        Logging level name (``INFO``, ``DEBUG``...).
    log_file:
        Optional path to a file for log output. When provided a ``FileHandler`` is
        attached in addition to the default stream handler.
    """

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if log_file:
        file_handler = logging.FileHandler(Path(log_file))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


__all__ = ["configure_logging"]
