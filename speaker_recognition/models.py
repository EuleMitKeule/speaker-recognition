"""Data models for speaker recognition."""

from pydantic import BaseModel, Field

from speaker_recognition.const import (
    DEFAULT_ACCESS_LOG,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
)


class Config(BaseModel):
    """Application configuration."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    access_log: bool = DEFAULT_ACCESS_LOG
    embeddings_directory: str = DEFAULT_EMBEDDINGS_DIR

    class ConfigDict:
        """Pydantic configuration."""

        validate_assignment = True


class AudioInput(BaseModel):
    """Audio input data model."""

    audio_data: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")


class VoiceSample(BaseModel):
    """Voice sample data model."""

    user: str = Field(..., description="User identifier")
    audio: AudioInput = Field(..., description="Audio input for voice sample")


class TrainingRequest(BaseModel):
    """Training request data model."""

    voice_samples: list[VoiceSample] = Field(..., description="List of voice samples")


class TrainingResult(BaseModel):
    """Result of training operation."""

    status: str
    trained_users: list[str]
    count: int


class RecognitionRequest(BaseModel):
    """Recognition request data model."""

    audio: AudioInput = Field(..., description="Audio input for recognition")


class RecognitionResult(BaseModel):
    """Result of recognition operation."""

    user_id: str
    confidence: float
    all_scores: dict[str, float]


class HealthResponse(BaseModel):
    """Health check response data model."""

    status: str


class ErrorResponse(BaseModel):
    """Error response data model."""

    error: str


config = Config()
