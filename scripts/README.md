# Development Scripts

This directory contains scripts for local development and testing of the IHP ML Models add-on.

## Scripts

### `develop.sh`
Start the local development environment with Home Assistant and the addon.

```bash
./scripts/develop.sh
```

This script:
- Creates test configuration directories
- Initializes a Home Assistant test instance
- Builds and starts the addon container
- Displays useful information and commands

### `test-addon.sh`
Run automated API tests against the running addon.

```bash
./scripts/test-addon.sh
```

Tests include:
- Health check endpoint
- Status endpoint
- Training with fake data
- Prediction endpoint
- Model listing

### `clean.sh`
Clean up all test data and containers.

```bash
./scripts/clean.sh
```

This removes:
- Docker containers and volumes
- Test configuration directory
- Test data directory
- Environment file

## Usage

1. **Start development**: `./scripts/develop.sh`
2. **Run tests**: `./scripts/test-addon.sh`
3. **View logs**: `docker-compose logs -f ihp-ml-addon`
4. **Clean up**: `./scripts/clean.sh`

See [DEVELOPMENT.md](../DEVELOPMENT.md) for detailed documentation.
