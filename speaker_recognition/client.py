"""Client for speaker recognition API."""

from typing import Optional

import httpx

from speaker_recognition.models import (
    HealthResponse,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
)


class SpeakerRecognitionClient:
    """Client for interacting with the speaker recognition API."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize the client.

        Args:
            base_url: Base URL of the speaker recognition API
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "SpeakerRecognitionClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def health_check(self) -> HealthResponse:
        """Check the health of the API.

        Returns:
            Health response indicating service status

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        client = self._ensure_client()
        response = await client.get("/health")
        response.raise_for_status()
        return HealthResponse(**response.json())

    async def train(self, request: TrainingRequest) -> TrainingResult:
        """Train the speaker recognition model.

        Args:
            request: Training request containing voice samples

        Returns:
            Training result with status and trained users

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        client = self._ensure_client()
        response = await client.post(
            "/train",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return TrainingResult(**response.json())

    async def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        """Recognize a speaker from audio data.

        Args:
            request: Recognition request containing audio data

        Returns:
            Recognition result with identified user and confidence

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        client = self._ensure_client()
        response = await client.post(
            "/recognize",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return RecognitionResult(**response.json())


class SyncSpeakerRecognitionClient:
    """Synchronous client for interacting with the speaker recognition API."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize the synchronous client.

        Args:
            base_url: Base URL of the speaker recognition API
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "SyncSpeakerRecognitionClient":
        """Enter context manager."""
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        if self._client:
            self._client.close()
            self._client = None

    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None

    def _ensure_client(self) -> httpx.Client:
        """Ensure client is initialized."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def health_check(self) -> HealthResponse:
        """Check the health of the API.

        Returns:
            Health response indicating service status

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        client = self._ensure_client()
        response = client.get("/health")
        response.raise_for_status()
        return HealthResponse(**response.json())

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Train the speaker recognition model.

        Args:
            request: Training request containing voice samples

        Returns:
            Training result with status and trained users

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        client = self._ensure_client()
        response = client.post(
            "/train",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return TrainingResult(**response.json())

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        """Recognize a speaker from audio data.

        Args:
            request: Recognition request containing audio data

        Returns:
            Recognition result with identified user and confidence

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        client = self._ensure_client()
        response = client.post(
            "/recognize",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return RecognitionResult(**response.json())
