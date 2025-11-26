"""Unit tests for Home Assistant History Reader adapter."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from infrastructure.adapters.ha_history_reader import HomeAssistantHistoryReader


class TestHomeAssistantHistoryReader:
    """Test the HA history reader adapter."""

    def test_initialization_with_supervisor_env(self):
        """Test initialization with supervisor environment variables."""
        with patch.dict('os.environ', {
            'SUPERVISOR_URL': 'http://supervisor/core',
            'SUPERVISOR_TOKEN': 'test_token_123'
        }):
            reader = HomeAssistantHistoryReader()
            assert reader._ha_url == 'http://supervisor/core'
            assert reader._ha_token == 'test_token_123'

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://homeassistant:8123',
            ha_token='custom_token',
            timeout=60
        )
        assert reader._ha_url == 'http://homeassistant:8123'
        assert reader._ha_token == 'custom_token'
        assert reader._timeout == 60

    @pytest.mark.asyncio
    async def test_is_available_constructs_correct_url(self):
        """Test that is_available constructs the correct URL with urljoin fix."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = await reader.is_available()

            # Verify the correct URL was used (should be /core/api/, not just /api/)
            called_url = mock_get.call_args[0][0]
            assert called_url == 'http://supervisor/core/api/'
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_handles_trailing_slash(self):
        """Test that is_available works with URLs that already have trailing slash."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core/',  # Already has trailing slash
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = await reader.is_available()

            # Should still produce correct URL
            called_url = mock_get.call_args[0][0]
            assert called_url == 'http://supervisor/core/api/'
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(self):
        """Test that is_available returns False on connection error."""
        import requests
        
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            result = await reader.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_non_200(self):
        """Test that is_available returns False on non-200 status."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_get.return_value = mock_response

            result = await reader.is_available()
            assert result is False

    def test_get_headers_includes_bearer_token(self):
        """Test that headers include proper authorization."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='my_secret_token'
        )

        headers = reader._get_headers()
        assert headers['Authorization'] == 'Bearer my_secret_token'
        assert headers['Content-Type'] == 'application/json'
