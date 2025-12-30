"""Logging configuration for speaker recognition service."""

import logging


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors matching uvicorn style."""

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"

    COLORS = {
        logging.DEBUG: blue,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelno, self.grey)

        # Format timestamp without microseconds
        log_time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Use module name without __main__
        logger_name = record.name
        if logger_name == "__main__":
            logger_name = "speaker-recognition"
        elif logger_name.startswith("speaker_recognition."):
            logger_name = f"speaker-recognition.{logger_name.split('.', 1)[1]}"
        elif logger_name == "uvicorn.error":
            logger_name = "uvicorn"

        # Build formatted message
        level_name = record.levelname
        message = record.getMessage()

        formatted = (
            f"{self.grey}{log_time}{self.reset} "
            f"{log_color}{level_name:8}{self.reset} "
            f"{self.cyan}{logger_name:23}{self.reset} "
            f"{message}"
        )

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def configure_logging(log_level: str) -> None:
    """Configure logging for the application.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Configure speaker_recognition loggers to use speaker-recognition prefix
    for logger_name in ["speaker_recognition", "__main__"]:
        app_logger = logging.getLogger(logger_name)
        app_logger.handlers.clear()
        app_logger.addHandler(handler)
        app_logger.setLevel(log_level.upper())
        app_logger.propagate = False

    # Configure uvicorn loggers with the same colored formatter
    for uvicorn_logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.setLevel(log_level.upper())
        uvicorn_logger.propagate = False
