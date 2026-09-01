#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"
SPLIT_JSON="${C6_SPLIT_JSON:-$REPO_ROOT/c6_split.json}"
SPLIT_SEED="${C6_SPLIT_SEED:-20260831}"

if [[ -f "$SPLIT_JSON" ]]; then
    echo "Reusing existing C6 split: $SPLIT_JSON"
    exit 0
fi

cd "$REPO_ROOT/src"
python c6_pilot.py split \
    --human-dir "$HUMAN_DIR" \
    --output "$SPLIT_JSON" \
    --seed "$SPLIT_SEED"
