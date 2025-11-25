"""Pytest configuration for IHP ML Models tests.

This module configures the Python path for tests to find the application modules.
"""

import sys
from pathlib import Path

# Add the application directory to the Python path for test imports
APP_DIR = Path(__file__).parent.parent / "ihp_ml_addon" / "rootfs" / "app"
sys.path.insert(0, str(APP_DIR))
