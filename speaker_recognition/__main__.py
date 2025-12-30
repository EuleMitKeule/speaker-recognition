"""Main entry point for speaker recognition service."""

import logging
import warnings

import typer
import uvicorn
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

from speaker_recognition.api import app  # noqa: E402
from speaker_recognition.const import (  # noqa: E402
    DEFAULT_ACCESS_LOG,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    ENV_ACCESS_LOG,
    ENV_EMBEDDINGS_DIR,
    ENV_HOST,
    ENV_LOG_LEVEL,
    ENV_PORT,
)
from speaker_recognition.logging_config import configure_logging  # noqa: E402
from speaker_recognition.models import config  # noqa: E402
from speaker_recognition.recognizer import recognizer  # noqa: E402

load_dotenv()
cli = typer.Typer(name="speaker-recognition", help="Speaker Recognition Service")

_LOGGER = logging.getLogger(__name__)


@cli.command()
def serve(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-h",
        help="Host to bind the server to",
        envvar=ENV_HOST,
    ),
    port: int = typer.Option(
        DEFAULT_PORT,
        "--port",
        "-p",
        help="Port to bind the server to",
        envvar=ENV_PORT,
    ),
    log_level: str = typer.Option(
        DEFAULT_LOG_LEVEL,
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        envvar=ENV_LOG_LEVEL,
    ),
    access_log: bool = typer.Option(
        DEFAULT_ACCESS_LOG,
        "--access-log/--no-access-log",
        help="Enable or disable access logging",
        envvar=ENV_ACCESS_LOG,
    ),
    embeddings_dir: str = typer.Option(
        DEFAULT_EMBEDDINGS_DIR,
        "--embeddings-dir",
        "-e",
        help="Directory to store voice embeddings",
        envvar=ENV_EMBEDDINGS_DIR,
    ),
) -> None:
    """Start the Speaker Recognition Service."""

    config.host = host
    config.port = port
    config.log_level = log_level.upper()
    config.access_log = access_log
    config.embeddings_directory = embeddings_dir

    recognizer.embeddings_directory = config.embeddings_directory

    configure_logging(config.log_level)

    _LOGGER.info("Starting Speaker Recognition Service...")
    _LOGGER.info(f"Host: {config.host}")
    _LOGGER.info(f"Port: {config.port}")
    _LOGGER.info(f"Log Level: {config.log_level}")
    _LOGGER.info(f"Embeddings Directory: {config.embeddings_directory}")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=config.access_log,
        log_config=None,
    )


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
