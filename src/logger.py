"""Centralized logging: console + rotating file handler, configured once."""
import logging
import logging.handlers

from src.constants import LOGS_DIR

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_configured = False


def _configure(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("demandiq")
    root.setLevel(level)
    fmt = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_h = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "demandiq.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger; configures handlers on first call."""
    _configure()
    return logging.getLogger(f"demandiq.{name}")
