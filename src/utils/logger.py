"""
Logging configuration for the POSAS-MLLM study pipeline.

Produces timestamped, structured logs to both console and file.
Every log entry includes ISO-8601 timestamps for the audit trail.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "posas",
    log_file: str = "outputs/pipeline.log",
    level: str = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (use 'posas' for the main pipeline).
        log_file: Path to the log file.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        console_output: Whether to also print to console.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
