"""日志配置：控制台 + 文件双通道，供 CLI 与 Streamlit 复用。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path | str | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("insight")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def shutdown_logging() -> None:
    """关闭日志文件句柄，避免 Windows 下临时目录/文件被占用。"""
    logger = logging.getLogger("insight")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001
            pass
