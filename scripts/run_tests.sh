#!/usr/bin/env bash
# scripts/run_tests.sh
# Run pytest for the Dataset Builder API endpoints.
#
# Usage:
#   ./scripts/run_tests.sh                                 # Run all tests
#   ./scripts/run_tests.sh --verbose                       # Verbose output
#   ./scripts/run_tests.sh -k Export                       # Only tests matching "Export"
#   ./scripts/run_tests.sh -k "TestEntries or TestExport"  # Only run a specific test class
#   ./scripts/run_tests.sh --cov                           # With coverage report
#   ./scripts/run_tests.sh --last-failed                   # Re-run last failed tests
#
#
#   TEST_PDF="Bulletin75-2026_en.pdf" ./scripts/run_tests.sh  # Use real PDF
#
# Fast tests only (no LLM)
#   ./scripts/run_tests.sh -k "not LLM and not E2EProcessing and not EntryEditing and not ExportWithReal and not PaginationWithReal and not ProgressTracking and not MultipleSources and not CancelDuring and not FullWorkflow"
#
# Just the full E2E workflow
#   ./scripts/run_tests.sh -k "TestFullWorkflowE2E"
#


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


# ── Configurable test PDF ────────────────────────────────────────
# If TEST_PDF is set, copy it to the test temp dir for integration tests.
# Leave empty to skip real-PDF tests (faster, no LLM calls needed).
TEST_PDF="${TEST_PDF:-}"

echo "========================================"
echo " Dataset Builder — Test Runner"
echo " Project root: $PROJECT_ROOT"
if [ -n "$TEST_PDF" ]; then
    echo " Test PDF:   $TEST_PDF"
else
    echo " Test PDF:   (none — skipping real PDF tests)"
fi
echo "========================================"


# ── Ensure virtual environment is active ────────────────────────
if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -f "$PROJECT_ROOT/env/bin/activate" ]; then
    echo ""
    echo "⚠️  No virtual environment detected."
    echo "   If you have one, activate it first:"
    echo "     source env/bin/activate"
    echo ""
    exit 1
fi

# ── Install test dependencies if needed ─────────────────────────
if ! python -c "import pytest" 2>/dev/null; then
    echo ""
    echo "📦 Installing pytest and dev dependencies..."
    pip install --quiet pytest pytest-asyncio httpx pyyaml
fi

# ── Validate config.yaml exists ─────────────────────────────────
if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
    echo ""
    echo "❌ config.yaml not found in project root!"
    exit 1
fi

echo ""
echo "✅ config.yaml found."

# ── Check if server is running ──────────────────────────────────
# Read host/port from config.yaml using python
SERVER_HOST=$(python -c "import yaml; c=yaml.safe_load(open('$PROJECT_ROOT/config.yaml')); print(c['server']['host'])")
SERVER_PORT=$(python -c "import yaml; c=yaml.safe_load(open('$PROJECT_ROOT/config.yaml')); print(c['server']['port'])")

# Normalize 0.0.0.0 to localhost for health check
if [ "$SERVER_HOST" = "0.0.0.0" ]; then
    CHECK_HOST="localhost"
else
    CHECK_HOST="$SERVER_HOST"
fi

echo ""
echo "🔍 Checking server at http://$CHECK_HOST:$SERVER_PORT ..."

if ! curl -sf "http://$CHECK_HOST:$SERVER_PORT/" > /dev/null 2>&1; then
    echo ""
    echo "❌ Server is NOT running at http://$CHECK_HOST:$SERVER_PORT"
    echo ""
    echo "   Start it first:"
    echo "     cd $PROJECT_ROOT"
    echo "     python main.py"
    echo ""
    exit 1
fi

echo "✅ Server is running and responsive."

# ── Build pytest arguments ──────────────────────────────────────
PYTEST_ARGS=(
    "$PROJECT_ROOT/tests"
    --tb=short
    --strict-markers
)

# Handle --verbose flag anywhere in arguments → maps to -vvv --capture=no
if printf -- '%s\n' "$@" | grep -q -- '--verbose'; then
    PYTEST_ARGS+=(-vvv --capture=no)
fi

# Pass ALL arguments through directly (preserves -k "pattern", etc.)
if [ $# -gt 0 ]; then
    PYTEST_ARGS+=("$@")
fi

# Default to -v only if no verbosity flag provided at all
if ! printf -- '%s\n' "$@" | grep -qE '(-v|(--verbose|--capture))'; then
    PYTEST_ARGS+=(-v)
fi


# ── Run pytest ──────────────────────────────────────────────────
echo ""
python -m pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All integration tests passed!"
else
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
