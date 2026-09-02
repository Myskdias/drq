#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"
RUNS_DIR="${C6_RUNS_DIR:-$REPO_ROOT/runs}"

AVAILABLE_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
MAX_PROCESSES="${C6_MAX_PROCESSES:-96}"
N_PROCESSES="${N_PROCESSES:-$AVAILABLE_CPUS}"
if (( N_PROCESSES > AVAILABLE_CPUS )); then
    N_PROCESSES="$AVAILABLE_CPUS"
fi
if (( N_PROCESSES > MAX_PROCESSES )); then
    N_PROCESSES="$MAX_PROCESSES"
fi

TIMEOUT="${C6_FOLLOWUP_TIMEOUT:-7200}"
MECHANISM_ROUND="${C6_MECHANISM_ROUND:-4}"

MODE="both"
if [[ $# -gt 0 && "$1" =~ ^(fixed|mechanism|both)$ ]]; then
    MODE="$1"
    shift
fi

case "$MODE" in
    fixed)
        MODE_ARGS=(--fixed-only)
        DEFAULT_ROOTS=(0 1 2 3)
        ;;
    mechanism)
        MODE_ARGS=(--mechanism-only)
        DEFAULT_ROOTS=(0 1 2)
        ;;
    both)
        MODE_ARGS=()
        DEFAULT_ROOTS=(0 1 2)
        ;;
    *)
        echo "Mode must be fixed, mechanism, or both" >&2
        exit 2
        ;;
esac

if [[ $# -eq 0 ]]; then
    ROOT_INDICES=("${DEFAULT_ROOTS[@]}")
else
    ROOT_INDICES=("$@")
fi

cd "$REPO_ROOT/src"

for root_index in "${ROOT_INDICES[@]}"; do
    if ! [[ "$root_index" =~ ^[0-9]+$ ]]; then
        echo "Root indices must be non-negative integers" >&2
        exit 2
    fi

    run_dir="$RUNS_DIR/c6_root_$root_index"
    if [[ \
        ! -f "$run_dir/args.pkl" \
        || ! -f "$run_dir/all_rounds_map_elites.pkl" \
        || ! -f "$run_dir/c6_rerank.json" \
    ]]; then
        echo "Incomplete or missing C6 data: $run_dir" >&2
        exit 2
    fi

    echo "=== C6 follow-up mode=$MODE root=$root_index ==="
    python c6_followup.py \
        --run-dir "$run_dir" \
        --human-dir "$HUMAN_DIR" \
        --n-processes "$N_PROCESSES" \
        --timeout "$TIMEOUT" \
        --mechanism-round "$MECHANISM_ROUND" \
        "${MODE_ARGS[@]}"
done
