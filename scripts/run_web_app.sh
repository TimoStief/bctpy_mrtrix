#!/bin/bash
# Launch the BCT Analysis Web Interface

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "🧠 BCT Analysis Web Interface"
echo "=========================================="

# Check if venv exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "📝 Please run: bash scripts/setup_env.sh"
    exit 1
fi

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"

# Navigate to web_app
cd "$PROJECT_ROOT/web_app"

# Install Flask and Waitress if needed
echo "Checking dependencies..."
python -m pip install flask waitress > /dev/null 2>&1

echo "✓ Starting web interface..."
echo ""

# Run the Flask app
python app.py
