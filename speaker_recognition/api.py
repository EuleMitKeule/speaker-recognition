"""FastAPI application for speaker recognition service."""

import logging

from fastapi import FastAPI, HTTPException

from speaker_recognition.models import (
    ErrorResponse,
    HealthResponse,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
)
from speaker_recognition.recognizer import recognizer

_LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Speaker Recognition Service",
    description="API for training and recognizing speakers using voice samples",
    version="0.0.0-dev",
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.post(
    "/train",
    response_model=TrainingResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Training"],
)
async def train(request: TrainingRequest) -> TrainingResult:
    """Train the speaker recognition model."""
    try:
        return recognizer.train(request)

    except ValueError as error:
        _LOGGER.error(f"Validation error during training: {error}")
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        _LOGGER.error(f"Error during training: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.post(
    "/recognize",
    response_model=RecognitionResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Recognition"],
)
async def recognize(request: RecognitionRequest) -> RecognitionResult:
    """Recognize speaker from audio data."""
    try:
        return recognizer.recognize(request)

    except (ValueError, RuntimeError) as error:
        _LOGGER.error(f"Recognition error: {error}")
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        _LOGGER.error(f"Error during recognition: {error}")
        raise HTTPException(status_code=500, detail=str(error))
