"""Tests for shared audio helper utilities."""

from __future__ import annotations

import base64
from pathlib import Path
import wave

import pytest

from speaker_recognition.audio import (
    LOCAL_MEDIA_PREFIX,
    media_selector_content_ids,
    read_wav_pcm,
    resolve_local_media_path,
)


def _write_wav(
    path: Path,
    pcm_data: bytes,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


def test_media_selector_content_ids_accepts_single_dict() -> None:
    """Extract content ID from the single-file selector shape."""
    assert media_selector_content_ids(
        {"media_content_id": "media-source://media_source/local/sample.wav"}
    ) == ["media-source://media_source/local/sample.wav"]


def test_media_selector_content_ids_accepts_list() -> None:
    """Extract content IDs from the multi-file selector shape."""
    assert media_selector_content_ids(
        [
            {"media_content_id": "media-source://media_source/local/one.wav"},
            {"media_content_id": "media-source://media_source/local/two.wav"},
        ]
    ) == [
        "media-source://media_source/local/one.wav",
        "media-source://media_source/local/two.wav",
    ]


def test_resolve_local_media_path(tmp_path: Path) -> None:
    """Resolve Home Assistant local media IDs under the media directory."""
    media_dir = tmp_path / "media"
    media_path = media_dir / "voices" / "ilya.wav"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.touch()

    assert resolve_local_media_path(
        f"{LOCAL_MEDIA_PREFIX}voices/ilya.wav", media_dir
    ) == media_path.resolve()


def test_resolve_local_media_path_checks_multiple_roots(tmp_path: Path) -> None:
    """Resolve local media from the first existing candidate root."""
    missing_media_dir = tmp_path / "config" / "media"
    media_dir = tmp_path / "media"
    media_path = media_dir / "ilya_voice_sample.wav"
    media_path.parent.mkdir(parents=True)
    media_path.touch()

    assert resolve_local_media_path(
        f"{LOCAL_MEDIA_PREFIX}ilya_voice_sample.wav",
        [missing_media_dir, media_dir],
    ) == media_path.resolve()


def test_resolve_local_media_path_reports_checked_paths(tmp_path: Path) -> None:
    """Report all checked local media paths when the file is missing."""
    with pytest.raises(FileNotFoundError, match="config.*media.*media"):
        resolve_local_media_path(
            f"{LOCAL_MEDIA_PREFIX}missing.wav",
            [tmp_path / "config" / "media", tmp_path / "media"],
        )


def test_resolve_local_media_path_rejects_path_escape(tmp_path: Path) -> None:
    """Reject media IDs that escape the configured media directory."""
    with pytest.raises(ValueError, match="escapes"):
        resolve_local_media_path(f"{LOCAL_MEDIA_PREFIX}../secrets.yaml", tmp_path)


def test_read_wav_pcm_returns_base64_pcm_and_sample_rate(tmp_path: Path) -> None:
    """Read WAV frames, not the WAV container bytes."""
    wav_path = tmp_path / "sample.wav"
    pcm_data = b"\x01\x00\x02\x00\x03\x00"
    _write_wav(wav_path, pcm_data, sample_rate=22050)

    prepared = read_wav_pcm(wav_path)

    assert prepared.audio_data == base64.b64encode(pcm_data).decode("utf-8")
    assert prepared.sample_rate == 22050


def test_read_wav_pcm_rejects_stereo(tmp_path: Path) -> None:
    """Reject unsupported multi-channel WAV input."""
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, b"\x00\x00\x00\x00", channels=2)

    with pytest.raises(ValueError, match="mono"):
        read_wav_pcm(wav_path)
