"""Speaker Recognition Service Package."""

from speaker_recognition.client import (
    SpeakerRecognitionClient,
    SyncSpeakerRecognitionClient,
)
from speaker_recognition.models import RecognitionResult, TrainingResult

__all__ = [
    "RecognitionResult",
    "SpeakerRecognitionClient",
    "SyncSpeakerRecognitionClient",
    "TrainingResult",
]
