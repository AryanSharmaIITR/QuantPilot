"""Structured logging for QuantPilot.

Use ``get_logger(__name__)`` in every module instead of ``print``. The level is
driven by ``logging.level`` in config.yaml and can be overridden per-call.
"""
from __future__ import annotations

import logging
import sys

from config import CONFIG

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "quantpilot", level: str | None = None) -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured — avoid duplicate handlers
        return logger

    level = (level or CONFIG.get("logging", {}).get("level", "INFO")).upper()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
