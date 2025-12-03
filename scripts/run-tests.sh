#!/bin/bash
# Quick test runner with common options

set -e

cd "$(dirname "$0")/.."

echo "🧪 IHP-ML-Models Test Runner"
echo "=============================="
echo ""

# Parse command
case "${1:-all}" in
    unit)
        echo "▶️  Running unit tests only..."
        poetry run pytest tests/unit/ -v
        ;;
    integration)
        echo "▶️  Running integration tests only..."
        poetry run pytest tests/integration/ -v
        ;;
    fast)
        echo "▶️  Running quick integration tests (no verbose)..."
        poetry run pytest tests/integration/ -q
        ;;
    coverage)
        echo "▶️  Running tests with coverage report..."
        poetry run pytest tests/integration/ --cov=ihp_ml_addon/rootfs/app --cov-report=term-missing --cov-report=html
        echo ""
        echo "📊 Coverage report generated in htmlcov/index.html"
        ;;
    ci)
        echo "▶️  Running tests in CI mode (strict)..."
        poetry run pytest tests/ -v --tb=short --strict-markers
        ;;
    watch)
        echo "▶️  Running tests in watch mode..."
        echo "   (re-runs tests on file changes)"
        poetry run pytest-watch tests/integration/ -v
        ;;
    debug)
        echo "▶️  Running tests in debug mode (verbose + full traceback)..."
        poetry run pytest tests/integration/ -vv --tb=long --log-cli-level=DEBUG
        ;;
    all|*)
        echo "▶️  Running all tests..."
        poetry run pytest tests/ -v
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Tests completed successfully!"
else
    echo ""
    echo "❌ Tests failed with exit code $EXIT_CODE"
fi

echo ""
echo "💡 Available commands:"
echo "   ./scripts/run-tests.sh unit          - Run unit tests only"
echo "   ./scripts/run-tests.sh integration   - Run integration tests only"
echo "   ./scripts/run-tests.sh fast          - Quick integration test run"
echo "   ./scripts/run-tests.sh coverage      - With coverage report"
echo "   ./scripts/run-tests.sh ci            - CI mode (strict)"
echo "   ./scripts/run-tests.sh debug         - Debug mode (verbose)"
echo "   ./scripts/run-tests.sh all           - All tests (default)"

exit $EXIT_CODE
