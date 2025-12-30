"""Pytest configuration for speaker recognition tests."""

import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure event loop policy for tests."""
    import asyncio

    return asyncio.get_event_loop_policy()
