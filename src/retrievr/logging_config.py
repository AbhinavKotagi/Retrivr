"""Logging bootstrap for Retrievr."""

from __future__ import annotations

import logging

from retrievr.config import settings


def configure_logging() -> None:
    """Configure process-wide logging once."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
