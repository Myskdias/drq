#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/c6_prepare.sh"
"$SCRIPT_DIR/run_c6_root.sh" 0
"$SCRIPT_DIR/run_c6_root.sh" 1
"$SCRIPT_DIR/c6_postprocess.sh"
