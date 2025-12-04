"""Adjacency configuration loader.

Manages the room topology configuration that defines which zones are adjacent
to each other for multi-room feature engineering.
"""

import json
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


class AdjacencyConfig:
    """Loads and manages room adjacency topology configuration."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize adjacency configuration.

        Args:
            config_path: Path to the adjacencies JSON file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/adjacencies_room.json relative to project root
            # Try to find it from the app directory
            app_dir = Path(__file__).parent.parent.parent
            config_path = app_dir.parent.parent.parent / "config" / "adjacencies_room.json"
        
        self._config_path = Path(config_path)
        self._adjacencies: dict[str, list[str]] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load the adjacency configuration from disk."""
        if not self._config_path.exists():
            _LOGGER.warning(
                "Adjacency config file not found at %s. Multi-room features will be disabled.",
                self._config_path
            )
            return

        try:
            with open(self._config_path, "r") as f:
                config_data = json.load(f)
            
            zones = config_data.get("zones", {})
            for zone_id, zone_config in zones.items():
                adjacent = zone_config.get("adjacent_zones", [])
                if not isinstance(adjacent, list):
                    _LOGGER.warning(
                        "Invalid adjacent_zones for zone %s, expected list, got %s",
                        zone_id, type(adjacent)
                    )
                    continue
                self._adjacencies[zone_id] = adjacent
            
            _LOGGER.info(
                "Loaded adjacency configuration for %d zones from %s",
                len(self._adjacencies), self._config_path
            )
        except (json.JSONDecodeError, OSError) as e:
            _LOGGER.error("Failed to load adjacency config from %s: %s", self._config_path, e)

    def get_adjacent_zones(self, zone_id: str) -> list[str]:
        """Get the list of adjacent zones for a given zone.

        Args:
            zone_id: The zone identifier

        Returns:
            List of adjacent zone IDs, or empty list if zone not configured
        """
        return self._adjacencies.get(zone_id, [])

    def has_adjacencies(self, zone_id: str) -> bool:
        """Check if a zone has any configured adjacencies.

        Args:
            zone_id: The zone identifier

        Returns:
            True if the zone has at least one adjacent zone
        """
        return zone_id in self._adjacencies and len(self._adjacencies[zone_id]) > 0

    def get_feature_names_for_zone(self, zone_id: str, base_features: tuple[str, ...]) -> tuple[str, ...]:
        """Generate the complete feature list for a zone including adjacent room features.

        Args:
            zone_id: The zone identifier
            base_features: Base features for the target zone

        Returns:
            Tuple of all feature names including adjacent room features
        """
        features = list(base_features)
        adjacent_zones = self.get_adjacent_zones(zone_id)
        
        # For each adjacent zone, add 4 features:
        # 1. current_temp
        # 2. current_humidity
        # 3. next_target_temp
        # 4. duration_until_change (in minutes)
        for adj_zone in adjacent_zones:
            features.extend([
                f"{adj_zone}_current_temp",
                f"{adj_zone}_current_humidity",
                f"{adj_zone}_next_target_temp",
                f"{adj_zone}_duration_until_change"
            ])
        
        return tuple(features)

    @property
    def all_zones(self) -> list[str]:
        """Get all configured zone IDs.

        Returns:
            List of all zone IDs in the configuration
        """
        return list(self._adjacencies.keys())
