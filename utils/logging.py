"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    output_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_filename: str = "run.log",
) -> logging.Logger:
    """Configure root logger with console + optional file handler.

    Args:
        output_dir: If provided, also log to a file in this directory.
        level: Logging level.
        log_filename: Name of the log file.

    Returns:
        The configured root logger.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if output_dir is not None:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("dreambooth_clora")
    logger.setLevel(level)
    return logger
