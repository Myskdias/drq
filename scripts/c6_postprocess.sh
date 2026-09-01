#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"
SPLIT_JSON="${C6_SPLIT_JSON:-$REPO_ROOT/c6_split.json}"
RUNS_DIR="${C6_RUNS_DIR:-$REPO_ROOT/runs}"
OUTPUT_DIR="${C6_RESULTS_DIR:-$REPO_ROOT/results/c6}"

AVAILABLE_CPUS="$(nproc)"
DEFAULT_PROCESSES="$AVAILABLE_CPUS"
if (( DEFAULT_PROCESSES > 20 )); then
    DEFAULT_PROCESSES=20
fi
N_PROCESSES="${N_PROCESSES:-$DEFAULT_PROCESSES}"
AUDIT_TIMEOUT="${C6_AUDIT_TIMEOUT:-21600}"

RUN_DIRS=("$RUNS_DIR/c6_root_0" "$RUNS_DIR/c6_root_1")
for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "$run_dir/args.pkl" || ! -f "$run_dir/all_rounds_map_elites.pkl" ]]; then
        echo "Incomplete or missing DRQ run: $run_dir" >&2
        exit 2
    fi
done

EXTRA_ARGS=()
if [[ "${C6_REUSE_AUDITS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--reuse-audits)
fi

cd "$REPO_ROOT/src"
python c6_postprocess.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --human-dir "$HUMAN_DIR" \
    --split-json "$SPLIT_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --n-processes "$N_PROCESSES" \
    --timeout "$AUDIT_TIMEOUT" \
    "${EXTRA_ARGS[@]}"

echo "C6 report outputs: $OUTPUT_DIR"
