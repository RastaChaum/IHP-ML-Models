# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed Home Assistant integration availability check - `urljoin()` was incorrectly removing `/core` from supervisor URL, causing API calls to fail at `http://supervisor/api/` instead of `http://supervisor/core/api/`. This prevented the `/api/v1/train/device` endpoint from working even though the addon was properly initialized with supervisor token.

## [0.1.0] - Initial Release

### Added
- XGBoost-based ML model for heating duration prediction
- Flask HTTP API with endpoints for training and prediction
- Home Assistant Supervisor integration for fetching historical data
- Fake data generator for testing without real sensor data
- Model persistence with file storage
- Multi-architecture Docker support (amd64, aarch64, armv7, armhf)
- Domain-Driven Design architecture with clear separation of concerns
