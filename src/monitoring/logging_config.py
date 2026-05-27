"""
Structured Logging Configuration for MLOps API.

Implements:
- JSON-formatted logs for machine-readable output
- Request ID tracing across services
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Separate log streams for application and access logs
- Audit logging for predictions and model operations
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pythonjsonlogger import jsonlogger

# Context variable for request ID (thread-safe)
request_id_var: ContextVar[str] = ContextVar("request_id", default="no-request-id")


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that includes:
    - Timestamp in ISO 8601 format
    - Request ID from context
    - Log level
    - Logger name
    - Message
    - Extra fields (if provided)
    """

    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record["timestamp"] = datetime.now(UTC).isoformat()

        # Add log level
        log_record["level"] = record.levelname

        # Add logger name
        log_record["logger"] = record.name

        # Add request ID from context
        log_record["request_id"] = request_id_var.get("no-request-id")

        # Add file and line number
        log_record["file"] = record.filename
        log_record["line"] = record.lineno

        # Add function name
        log_record["function"] = record.funcName

        # Add process and thread info
        log_record["process"] = record.process
        log_record["thread"] = record.thread

        # If exception info exists, add it
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
) -> None:
    """
    Set up structured JSON logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        enable_file_logging: Whether to write logs to files
        enable_console_logging: Whether to write logs to console
    """
    # Create log directory if needed
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create custom formatter
    formatter = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={
            "levelname": "level",
            "name": "logger",
            "pathname": "file",
            "lineno": "line",
            "funcName": "function",
        },
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Add console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handlers
    if enable_file_logging:
        # Application log (all messages)
        app_log_path = log_dir / "app.log"
        app_handler = logging.FileHandler(app_log_path)
        app_handler.setLevel(logging.DEBUG)  # Log everything to file
        app_handler.setFormatter(formatter)
        root_logger.addHandler(app_handler)

        # Error log (ERROR and above only)
        error_log_path = log_dir / "error.log"
        error_handler = logging.FileHandler(error_log_path)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Reduce noise
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def set_request_id(request_id: str | None = None) -> str:
    """
    Set request ID in context for current request.

    Args:
        request_id: Custom request ID or None to generate one

    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    request_id_var.set(request_id)
    return request_id


def get_request_id() -> str:
    """
    Get current request ID from context.

    Returns:
        Current request ID or 'no-request-id' if not set
    """
    return request_id_var.get("no-request-id")


def log_audit_event(
    event_type: str,
    user_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Log an audit event for compliance and security.

    Args:
        event_type: Type of event (e.g., 'prediction', 'model_reload', 'label_injection')
        user_id: User ID or API key that triggered the event
        details: Additional event details
    """
    logger = logging.getLogger("audit")

    audit_record = {
        "event_type": event_type,
        "user_id": user_id or "system",
        "request_id": get_request_id(),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if details:
        audit_record["details"] = details  # type: ignore[assignment]

    logger.info("audit_event", extra=audit_record)


def log_prediction(
    prediction_id: int,
    signal_id: int,
    prediction: int,
    confidence: float,
    user_id: str | None = None,
    model_version: str | None = None,
) -> None:
    """
    Log a prediction event for audit trail.

    Args:
        prediction_id: Database ID of prediction
        signal_id: Database ID of signal
        prediction: Predicted label (0 or 1)
        confidence: Prediction confidence score
        user_id: User making the prediction
        model_version: Model version used
    """
    log_audit_event(
        event_type="prediction",
        user_id=user_id,
        details={
            "prediction_id": prediction_id,
            "signal_id": signal_id,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
        },
    )


def log_model_operation(
    operation: str,
    model_version: str | None = None,
    registry_version: int | None = None,
    user_id: str | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
    """
    Log a model operation (reload, promotion, rollback) for audit trail.

    Args:
        operation: Type of operation (reload, promote, rollback)
        model_version: Model version string
        registry_version: MLflow registry version
        user_id: User performing the operation
        success: Whether operation succeeded
        error: Error message if operation failed
    """
    log_audit_event(
        event_type=f"model_{operation}",
        user_id=user_id,
        details={
            "model_version": model_version,
            "registry_version": registry_version,
            "success": success,
            "error": error,
        },
    )


def log_label_injection(
    signal_id: int,
    ground_truth: int,
    source: str,
    user_id: str | None = None,
) -> None:
    """
    Log a label injection event for audit trail.

    Args:
        signal_id: Database ID of signal
        ground_truth: Ground truth label (0 or 1)
        source: Source of label (manual, automated, import)
        user_id: User injecting the label
    """
    log_audit_event(
        event_type="label_injection",
        user_id=user_id,
        details={
            "signal_id": signal_id,
            "ground_truth": ground_truth,
            "source": source,
        },
    )


# Example usage in tests or scripts
if __name__ == "__main__":
    # Set up logging
    setup_logging(log_level="DEBUG")

    # Create logger
    logger = logging.getLogger(__name__)

    # Set request ID
    set_request_id("test-request-123")

    # Log some messages
    logger.debug("This is a debug message", extra={"user": "test_user"})
    logger.info("This is an info message", extra={"action": "test"})
    logger.warning("This is a warning message")
    logger.error("This is an error message", extra={"code": 500})

    # Log audit event
    log_audit_event(
        event_type="test_event",
        user_id="test_user",
        details={"test_key": "test_value"},
    )

    # Log prediction
    log_prediction(
        prediction_id=1,
        signal_id=100,
        prediction=0,
        confidence=0.95,
        user_id="test_user",
        model_version="2026-02-19",
    )

    print("Check logs/app.log for JSON-formatted output")
