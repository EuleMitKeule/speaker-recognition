"""Audio helpers shared by the API client and Home Assistant integration."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import wave

LOCAL_MEDIA_PREFIX = "media-source://media_source/local/"


@dataclass(frozen=True)
class PcmAudio:
    """Base64 encoded PCM audio prepared for the recognition API."""

    audio_data: str
    sample_rate: int


def media_selector_content_ids(value: object) -> list[str]:
    """Extract media content IDs from Home Assistant MediaSelector output."""
    if isinstance(value, dict):
        items: list[object] = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    content_ids: list[str] = []
    for item in items:
        media_content_id: object | None
        if isinstance(item, dict):
            media_content_id = item.get("media_content_id")
        else:
            media_content_id = item

        if isinstance(media_content_id, str) and media_content_id:
            content_ids.append(media_content_id)

    return content_ids


def resolve_local_media_path(
    media_content_id: str, media_dirs: Path | Iterable[Path]
) -> Path | None:
    """Resolve a Home Assistant local media source ID to a file path."""
    if not media_content_id.startswith(LOCAL_MEDIA_PREFIX):
        return None

    relative_path = media_content_id[len(LOCAL_MEDIA_PREFIX) :]
    if isinstance(media_dirs, Path):
        candidate_dirs = [media_dirs]
    else:
        candidate_dirs = list(media_dirs)

    checked_paths: list[Path] = []
    for media_dir in candidate_dirs:
        base_dir = media_dir.resolve()
        file_path = (base_dir / relative_path).resolve()

        try:
            file_path.relative_to(base_dir)
        except ValueError as error:
            raise ValueError("Media path escapes the local media directory") from error

        checked_paths.append(file_path)
        if file_path.exists():
            return file_path

    checked = ", ".join(str(path) for path in checked_paths)
    raise FileNotFoundError(f"Local media file not found. Checked: {checked}")


def read_wav_pcm(path: Path) -> PcmAudio:
    """Read a mono 16-bit PCM WAV file and return API-ready audio data."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

            if channels != 1:
                raise ValueError(
                    f"Only mono WAV files are supported, got {channels} channels"
                )
            if sample_width != 2:
                raise ValueError(
                    f"Only 16-bit PCM WAV files are supported, got {sample_width} bytes"
                )

            pcm_data = wav_file.readframes(frame_count)
    except wave.Error as error:
        raise ValueError(f"Invalid WAV file: {path}") from error

    if not pcm_data:
        raise ValueError(f"Empty WAV file: {path}")

    return PcmAudio(
        audio_data=base64.b64encode(pcm_data).decode("utf-8"),
        sample_rate=sample_rate,
    )
