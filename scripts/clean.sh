#!/bin/bash
# Clean up development environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Stopping containers..."
docker compose down -v

echo "Removing test data..."
rm -rf test-config test-data

echo "Removing .env file..."
rm -f .env

echo ""
echo "Development environment cleaned!"
echo "Run './scripts/develop.sh' to set up again."
