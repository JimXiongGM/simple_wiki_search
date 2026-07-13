from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from loguru import logger


def setup_loguru(
    *,
    level: str = "INFO",
    sink: TextIO | None = None,
    fmt: str | None = None,
    intercept_stdlib: bool = False,
    intercept_logger_names: Iterable[str] = (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ),
    root_level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    logger.remove()

    default_fmt = (
        fmt
        or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.add(
        sys.stderr if sink is None else sink,
        level=(level or "INFO").upper(),
        backtrace=False,
        diagnose=False,
        format=default_fmt,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        logger.add(
            str(log_path),
            level=(level or "INFO").upper(),
            backtrace=False,
            diagnose=False,
            format=file_fmt,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
        )

    if not intercept_stdlib:
        return

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "Terminating session" in msg:
                return
            if "Processing request of type" in msg:
                if any(
                    x in msg
                    for x in (
                        "CallToolRequest",
                        "ListToolsRequest",
                        "ListResourcesRequest",
                        "ListPromptsRequest",
                        "GetResourceRequest",
                    )
                ):
                    return
            try:
                lvl = logger.level(record.levelname).name
            except Exception:
                lvl = record.levelno
            if record.exc_info:
                logger.opt(exception=record.exc_info).log(lvl, msg)
                return
            if record.stack_info:
                logger.log(lvl, "{}\n{}", msg, record.stack_info)
                return
            logger.log(lvl, msg)

    root = logging.getLogger()
    root.handlers = [_InterceptHandler()]
    root.setLevel(root_level)

    for name in intercept_logger_names:
        l = logging.getLogger(name)
        l.handlers = [_InterceptHandler()]
        l.propagate = False
