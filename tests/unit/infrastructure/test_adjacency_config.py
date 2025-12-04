"""Tests for adjacency configuration loader."""

import json
import tempfile
from pathlib import Path

import pytest


def test_adjacency_config_load_from_file():
    """Test loading adjacency configuration from a JSON file."""
    from infrastructure.adapters import AdjacencyConfig

    # Create a temporary config file
    config_data = {
        "zones": {
            "living_room": {"adjacent_zones": ["kitchen", "hallway"]},
            "bedroom": {"adjacent_zones": ["hallway"]},
            "kitchen": {"adjacent_zones": ["living_room"]},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        config = AdjacencyConfig(config_path)

        # Test get_adjacent_zones
        assert config.get_adjacent_zones("living_room") == ["kitchen", "hallway"]
        assert config.get_adjacent_zones("bedroom") == ["hallway"]
        assert config.get_adjacent_zones("kitchen") == ["living_room"]
        assert config.get_adjacent_zones("unknown_zone") == []

        # Test has_adjacencies
        assert config.has_adjacencies("living_room") is True
        assert config.has_adjacencies("bedroom") is True
        assert config.has_adjacencies("unknown_zone") is False

        # Test all_zones
        assert set(config.all_zones) == {"living_room", "bedroom", "kitchen"}

    finally:
        Path(config_path).unlink()


def test_adjacency_config_missing_file():
    """Test handling of missing config file."""
    from infrastructure.adapters import AdjacencyConfig

    config = AdjacencyConfig("/nonexistent/path/config.json")

    # Should return empty results when config file doesn't exist
    assert config.get_adjacent_zones("any_zone") == []
    assert config.has_adjacencies("any_zone") is False
    assert config.all_zones == []


def test_adjacency_config_feature_names():
    """Test generation of feature names for a zone."""
    from infrastructure.adapters import AdjacencyConfig

    config_data = {
        "zones": {
            "living_room": {"adjacent_zones": ["kitchen", "hallway"]},
            "bedroom": {"adjacent_zones": []},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        config = AdjacencyConfig(config_path)
        base_features = ("outdoor_temp", "indoor_temp", "target_temp", "temp_delta")

        # Test zone with adjacent rooms
        feature_names = config.get_feature_names_for_zone("living_room", base_features)
        assert feature_names[:4] == base_features
        # Should have 4 features per adjacent zone (2 zones = 8 features)
        assert len(feature_names) == 4 + 8
        assert "kitchen_current_temp" in feature_names
        assert "kitchen_current_humidity" in feature_names
        assert "kitchen_next_target_temp" in feature_names
        assert "kitchen_duration_until_change" in feature_names
        assert "hallway_current_temp" in feature_names

        # Test zone without adjacent rooms
        feature_names = config.get_feature_names_for_zone("bedroom", base_features)
        assert feature_names == base_features

    finally:
        Path(config_path).unlink()


def test_adjacency_config_invalid_data():
    """Test handling of invalid configuration data."""
    from infrastructure.adapters import AdjacencyConfig

    # Create config with invalid adjacent_zones (not a list)
    config_data = {
        "zones": {
            "living_room": {"adjacent_zones": "not_a_list"},
            "kitchen": {"adjacent_zones": ["living_room"]},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        config = AdjacencyConfig(config_path)

        # Invalid zone should be skipped, valid zone should work
        assert config.get_adjacent_zones("living_room") == []
        assert config.get_adjacent_zones("kitchen") == ["living_room"]

    finally:
        Path(config_path).unlink()
