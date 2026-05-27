"""
Central loguru-based logging configuration for the Streamlit UI application.

Usage — in any view or component:

    from src.ui.logging_ui import get_ui_logger
    logger = get_ui_logger(__name__)

    logger.info("Page loaded")
    logger.warning("No containers found")
    logger.error("Database unreachable: {err}", err=exc)

Call ``configure_ui_logging()`` once, at the top of ``src/ui/app.py``,
before any page module is imported.  All subsequent ``get_ui_logger()``
calls in any view share the same loguru sink configuration.

Log file location
-----------------
Defaults to ``logs/ui_app.log`` relative to the project root.
Override by setting the ``UI_LOG_FILE`` environment variable (done
automatically by the ``make ui`` target).

The file is cleared (truncated) on every Streamlit restart so the console
starts with a clean slate, matching the Docker Operations Console behaviour.

Rotating file policy: 5 MB per file, up to 3 retained files.

Thread safety
-------------
Loguru's sinks are thread-safe by default; the file handler uses its own
internal lock.  Safe to use from Streamlit callbacks running on multiple
threads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# ── Module-level state ────────────────────────────────────────────────────────

_UI_LOG_FILE: Path | None = None
_CONFIGURED: bool = False

# ── Public API ─────────────────────────────────────────────────────────────────


def configure_ui_logging(log_file: str | None = None) -> None:
    """Set up loguru sinks for the Streamlit UI application.

    Must be called once at application startup (in ``app.py``, at module
    level, before any page is rendered).  Subsequent calls are no-ops.

    Sinks:
    - **RotatingFileHandler** → ``log_file`` (5 MB × 3 files, UTF-8)
    - **StreamHandler**      → ``sys.__stderr__`` (bypasses Streamlit's
      captured stderr; output appears in the terminal running ``make ui``)

    The log file is **cleared on startup** so the console starts fresh on
    every ``make ui`` invocation.

    Args:
        log_file: Absolute or relative path to the log file.  Falls back to
                  the ``UI_LOG_FILE`` environment variable, then
                  ``logs/ui_app.log`` relative to the current working
                  directory.
    """
    global _UI_LOG_FILE, _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # ── Resolve log file path ────────────────────────────────────────────────
    if log_file is None:
        log_file = os.environ.get("UI_LOG_FILE", "logs/ui_app.log")

    _UI_LOG_FILE = Path(log_file)
    _UI_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Clear on startup (clean slate, matches Docker Operations Console behaviour)
    _UI_LOG_FILE.write_text("", encoding="utf-8")

    # ── Remove loguru's default stderr handler ───────────────────────────────
    logger.remove()

    # ── File sink (rotating, persistent across browser reconnects) ───────────
    logger.add(
        str(_UI_LOG_FILE),
        rotation="5 MB",
        retention=3,
        level="DEBUG",
        format=("{time:YYYY-MM-DD HH:mm:ss}  {level:<8}  {name:<35}  {message}"),
        encoding="utf-8",
        enqueue=True,  # thread-safe async write
        backtrace=True,
        diagnose=False,  # disable variable values in tracebacks (security)
    )

    # ── Terminal sink (real stderr — bypasses Streamlit capture) ────────────
    logger.add(
        sys.__stderr__ or sys.stderr,  # type: ignore[arg-type]  # may be None in some envs
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green>  "
            "<level>{level:<8}</level>  "
            "<cyan>{name:<25}</cyan>  {message}"
        ),
        colorize=True,
        enqueue=False,
    )

    # ── Exception hook ───────────────────────────────────────────────────────
    # Catch any exception that propagates out of a thread before Streamlit's
    # own handler sees it.  Streamlit catches most exceptions internally and
    # shows an error banner, but they are never written to a file.
    _original_excepthook = sys.excepthook

    def _ui_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: object,
    ) -> None:
        import types

        tb = exc_tb if isinstance(exc_tb, types.TracebackType) else None
        logger.opt(exception=(exc_type, exc_value, tb)).critical("Unhandled exception")
        _original_excepthook(exc_type, exc_value, exc_tb)  # type: ignore[arg-type]

    sys.excepthook = _ui_excepthook

    logger.info("UI logging configured — log file: {}", _UI_LOG_FILE)


def get_ui_logger(name: str = "ui") -> Logger:
    """Return a loguru logger bound to *name*.

    If ``configure_ui_logging()`` has not yet been called (e.g. when running
    a module outside Streamlit), it is called automatically with default
    settings.

    Args:
        name: Typically ``__name__`` of the calling module.  Appears in the
              ``name`` field of every log record emitted by the returned
              logger.

    Returns:
        A loguru ``Logger`` instance that writes to the configured sinks.

    Example::

        from src.ui.logging_ui import get_ui_logger
        logger = get_ui_logger(__name__)
        logger.info("Rendering docker control page")
        logger.error("Stack start failed: rc={rc}", rc=rc)
    """
    if not _CONFIGURED:
        configure_ui_logging()
    # loguru's logger is a module-level singleton; bind() adds contextual
    # fields that appear in every record emitted by the returned logger.
    return logger.bind(name=name)


def get_log_file_path() -> Path | None:
    """Return the configured log file path, or ``None`` if not configured."""
    return _UI_LOG_FILE
