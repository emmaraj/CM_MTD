"""
Logging utility for CM-MTD framework.
Provides structured, color-coded logging to both console and file.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_loggers: dict = {}


def setup_logger(
    name: str = "cm_mtd",
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    log_file: bool = True,
) -> logging.Logger:
    """
    Set up a structured logger.
    
    Args:
        name: Logger name (used as identifier).
        log_dir: Directory to write log files.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        console: Whether to log to stdout.
        log_file: Whether to log to file.
    
    Returns:
        Configured Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"{name}_{timestamp}.log"
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _loggers[name] = logger
    return logger


def get_logger(name: str = "cm_mtd") -> logging.Logger:
    """Retrieve an existing logger or create a new one."""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]
