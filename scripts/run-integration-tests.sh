#!/bin/bash
# Script to run integration tests using Poetry

set -e

cd "$(dirname "$0")/.."

echo "🔧 Setting up test environment with Poetry..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install it first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies
echo "📥 Installing dependencies..."
poetry install --with dev,test --quiet

# Run tests
echo ""
echo "🧪 Running integration tests..."
echo "================================"
poetry run pytest tests/integration/ -v "$@"

# Check exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "💡 Tips:"
    echo "   - Run specific test: ./scripts/run-integration-tests.sh -k test_name"
    echo "   - With coverage: ./scripts/run-integration-tests.sh --cov"
    echo "   - Verbose: ./scripts/run-integration-tests.sh -vv"
else
    echo ""
    echo "❌ Some tests failed!"
fi

exit $EXIT_CODE
