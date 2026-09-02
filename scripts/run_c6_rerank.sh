#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"

RUNS_DIR="${C6_RUNS_DIR:-$REPO_ROOT/runs}"

AVAILABLE_CPUS="$(nproc)"
N_PROCESSES="${N_PROCESSES:-$AVAILABLE_CPUS}"

if (( N_PROCESSES > 20 )); then
    N_PROCESSES=20
fi

REPEATS="${C6_RERANK_REPEATS:-24}"
TIMEOUT="${C6_RERANK_TIMEOUT:-21600}"

if [[ $# -eq 0 ]]; then
    ROOT_INDICES=(0 1)
else
    ROOT_INDICES=("$@")
fi

cd "$REPO_ROOT/src"

for root_index in "${ROOT_INDICES[@]}"; do
    if ! [[ "$root_index" =~ ^[0-9]+$ ]]; then
        echo \
            "Root indices must be non-negative integers" \
            >&2
        exit 2
    fi

    run_dir="$RUNS_DIR/c6_root_$root_index"

    if [[ \
        ! -f "$run_dir/args.pkl" \
        || ! -f "$run_dir/all_rounds_map_elites.pkl" \
    ]]; then
        echo \
            "Incomplete or missing DRQ run: $run_dir" \
            >&2
        exit 2
    fi

    echo "=== C6 reranking root $root_index ==="

    python c6_rerank.py \
        --run-dir "$run_dir" \
        --human-dir "$HUMAN_DIR" \
        --n-processes "$N_PROCESSES" \
        --timeout "$TIMEOUT" \
        --repeats "$REPEATS"
done
