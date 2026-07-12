#!/bin/bash
# Wrapper pour le réentraînement hebdomadaire via cron.
# Adapter PROJECT_DIR au chemin réel de déploiement.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

python scripts/retrain.py >> "$LOG_DIR/retrain.log" 2>&1
