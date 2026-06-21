"""Настройка логирования для приложения."""

import logging
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
LOG_LEVEL_ENV = "MANIPULATOR_LOG_LEVEL"


def setup_logging(level: str | None = None) -> None:
    """Инициализирует корневой логгер для CLI-процессов."""
    level_name = (level or os.environ.get(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL)).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr,
        force=True,
    )

    logging.getLogger("manipulator_2d").setLevel(log_level)
