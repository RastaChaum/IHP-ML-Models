"""Pytest configuration for e2e tests."""

import pytest


def pytest_configure(config):
    """Register custom markers for e2e tests."""
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test requiring external systems",
    )
    config.addinivalue_line(
        "markers",
        "requires_ha: mark test as requiring a Home Assistant instance",
    )
