#!/usr/bin/env bash
# jarvis_v2 setup script
set -euo pipefail

echo "==> jarvis_v2 setup"

# Verify Python 3.13
if ! command -v python3.13 &>/dev/null; then
    echo "ERROR: python3.13 not found. Install via Homebrew: brew install python@3.13"
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "==> Creating venv with Python 3.13"
    python3.13 -m venv venv
fi

source venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip
pip install -e ".[dev]"

# Create logs directory
mkdir -p logs

# Copy .env if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "==> Created .env from .env.example — edit it with your credentials"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .env with your PostgreSQL, Neo4j, Redis, and Alpaca credentials"
echo "  2. Start all workers: supervisorctl -c config/supervisor.conf start all"
echo "  3. Or run the API: uvicorn jarvis.api.server:app --reload --port 8504"
