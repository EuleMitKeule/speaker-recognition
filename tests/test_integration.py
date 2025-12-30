"""Integration tests for speaker recognition API and client."""

import base64
import wave
from multiprocessing import Process
from pathlib import Path

import pytest
import uvicorn

from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import (
    AudioInput,
    RecognitionRequest,
    TrainingRequest,
    VoiceSample,
)

EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_data"
API_HOST = "127.0.0.1"
API_PORT = 8765
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


def start_api_server():
    """Start the API server in a subprocess."""
    from speaker_recognition.api import app

    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="error")


def read_audio_file_as_base64(file_path: Path) -> tuple[str, int]:
    """Read WAV file and encode PCM data as base64.

    Args:
        file_path: Path to the WAV audio file

    Returns:
        Tuple of (Base64 encoded PCM audio data, actual sample rate)
    """
    with wave.open(str(file_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm_data = wav_file.readframes(wav_file.getnframes())
        return base64.b64encode(pcm_data).decode("utf-8"), sample_rate


@pytest.fixture(scope="module")
def api_server():
    """Start API server for testing."""
    import time

    import httpx

    server_process = Process(target=start_api_server)
    server_process.start()

    # Wait for server to be ready with health checks
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = httpx.get(f"{API_BASE_URL}/health", timeout=1.0)
            if response.status_code == 200:
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt == max_attempts - 1:
                server_process.terminate()
                server_process.join(timeout=5)
                raise RuntimeError("Server failed to start within timeout")
            time.sleep(0.5)

    yield

    # Cleanup
    server_process.terminate()
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.kill()


@pytest.mark.asyncio
async def test_train_and_recognize_speakers(api_server: None):
    """Test training with two speakers and recognizing them with different samples."""
    # Read training audio files
    speaker1_training_file = EXAMPLE_DATA_DIR / "speaker1_1.wav"
    speaker2_training_file = EXAMPLE_DATA_DIR / "speaker2_1.wav"

    # Read recognition audio files
    speaker1_recognition_file = EXAMPLE_DATA_DIR / "speaker1_2.wav"
    speaker2_recognition_file = EXAMPLE_DATA_DIR / "speaker2_2.wav"

    # Verify all files exist
    assert speaker1_training_file.exists(), f"Missing {speaker1_training_file}"
    assert speaker2_training_file.exists(), f"Missing {speaker2_training_file}"
    assert speaker1_recognition_file.exists(), f"Missing {speaker1_recognition_file}"
    assert speaker2_recognition_file.exists(), f"Missing {speaker2_recognition_file}"

    async with SpeakerRecognitionClient(API_BASE_URL, timeout=60.0) as client:
        # Health check
        health = await client.health_check()
        assert health.status == "healthy"

        # Prepare training data
        speaker1_audio_data, speaker1_rate = read_audio_file_as_base64(
            speaker1_training_file
        )
        speaker2_audio_data, speaker2_rate = read_audio_file_as_base64(
            speaker2_training_file
        )

        training_request = TrainingRequest(
            voice_samples=[
                VoiceSample(
                    user="speaker1",
                    audio=AudioInput(
                        audio_data=speaker1_audio_data, sample_rate=speaker1_rate
                    ),
                ),
                VoiceSample(
                    user="speaker2",
                    audio=AudioInput(
                        audio_data=speaker2_audio_data, sample_rate=speaker2_rate
                    ),
                ),
            ]
        )

        # Train the model
        training_result = await client.train(training_request)
        assert training_result.status == "success"
        assert training_result.count == 2
        assert "speaker1" in training_result.trained_users
        assert "speaker2" in training_result.trained_users

        # Test recognition for speaker1
        speaker1_recognition_audio, speaker1_rec_rate = read_audio_file_as_base64(
            speaker1_recognition_file
        )
        recognition_request_1 = RecognitionRequest(
            audio=AudioInput(
                audio_data=speaker1_recognition_audio, sample_rate=speaker1_rec_rate
            )
        )

        recognition_result_1 = await client.recognize(recognition_request_1)
        assert recognition_result_1.user_id == "speaker1", (
            f"Expected speaker1, got {recognition_result_1.user_id} "
            f"with confidence {recognition_result_1.confidence}. "
            f"All scores: {recognition_result_1.all_scores}"
        )

        # Test recognition for speaker2
        speaker2_recognition_audio, speaker2_rec_rate = read_audio_file_as_base64(
            speaker2_recognition_file
        )
        recognition_request_2 = RecognitionRequest(
            audio=AudioInput(
                audio_data=speaker2_recognition_audio, sample_rate=speaker2_rec_rate
            )
        )

        recognition_result_2 = await client.recognize(recognition_request_2)
        assert recognition_result_2.user_id == "speaker2", (
            f"Expected speaker2, got {recognition_result_2.user_id} "
            f"with confidence {recognition_result_2.confidence}. "
            f"All scores: {recognition_result_2.all_scores}"
        )
