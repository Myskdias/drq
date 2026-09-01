#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 ROOT_INDEX" >&2
    exit 2
fi

ROOT_INDEX="$1"
if ! [[ "$ROOT_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ROOT_INDEX must be a non-negative integer" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"
SPLIT_JSON="${C6_SPLIT_JSON:-$REPO_ROOT/c6_split.json}"
RUNS_DIR="${C6_RUNS_DIR:-$REPO_ROOT/runs}"
N_ITERS="${N_ITERS:-250}"
N_ROUNDS=4
SIMULATION_SEED_STRIDE="${C6_SIMULATION_SEED_STRIDE:-10000}"
SIMULATION_SEED_OFFSET=$((ROOT_INDEX * SIMULATION_SEED_STRIDE))

AVAILABLE_CPUS="$(nproc)"
DEFAULT_PROCESSES="$AVAILABLE_CPUS"
if (( DEFAULT_PROCESSES > 20 )); then
    DEFAULT_PROCESSES=20
fi
N_PROCESSES="${N_PROCESSES:-$DEFAULT_PROCESSES}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set." >&2
    echo 'Run: export OPENAI_API_KEY="sk-..."' >&2
    exit 2
fi

"$REPO_ROOT/scripts/c6_prepare.sh"

INITIAL_OPPONENT="$(python - "$SPLIT_JSON" "$ROOT_INDEX" <<'PY'
import json
import sys

split_path, index = sys.argv[1], int(sys.argv[2])
with open(split_path) as f:
    initialization = json.load(f)["initialization"]
if index >= len(initialization):
    raise SystemExit(f"Root index {index} exceeds initialization pool size {len(initialization)}")
print(initialization[index])
PY
)"

RUN_DIR="$RUNS_DIR/c6_root_$ROOT_INDEX"
mkdir -p "$RUN_DIR"

cd "$REPO_ROOT/src"

COMMAND=(
    python drq.py
    --seed "$ROOT_INDEX"
    --llm-seed "$ROOT_INDEX"
    --simulation-seed-offset "$SIMULATION_SEED_OFFSET"
    --save-dir "$RUN_DIR"
    --resume True
    --job-timeout 604800
    --n-processes "$N_PROCESSES"
    --timeout 900
    --initial-opps "$HUMAN_DIR/$INITIAL_OPPONENT"
    --n-rounds "$N_ROUNDS"
    --n-iters "$N_ITERS"
    --log-every 20
    --last-k-opps 20
    --sample-new-percent 0.1
    --bc-axes "tsp,mc"
    --warmup-with-init-opps True
    --warmup-with-past-champs True
    --n-init 8
    --n-mutate 1
    --fitness-threshold 999
    --gpt-model "gpt-4.1-mini-2025-04-14"
    --llm-backend openai
    --temperature 1.0
    --simargs.rounds 20
    --simargs.size 8000
    --simargs.cycles 80000
    --simargs.processes 8000
    --simargs.length 100
    --simargs.distance 100
    --system-prompt "./prompts/system_prompt_0.txt"
    --new-prompt "./prompts/new_prompt_0.txt"
    --mutate-prompt "./prompts/mutate_prompt_0.txt"
)

{
    printf 'root_index=%s\n' "$ROOT_INDEX"
    printf 'initial_opponent=%s\n' "$INITIAL_OPPONENT"
    printf 'n_iters=%s\n' "$N_ITERS"
    printf 'n_processes=%s\n' "$N_PROCESSES"
    printf 'simulation_seed_offset=%s\n' "$SIMULATION_SEED_OFFSET"
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "$RUN_DIR/run_config.txt"

echo "=== C6 root $ROOT_INDEX ==="
echo "Initial opponent: $INITIAL_OPPONENT"
echo "Iterations per round: $N_ITERS"
echo "CPU workers: $N_PROCESSES"
echo "Output: $RUN_DIR"

"${COMMAND[@]}" 2>&1 | tee -a "$RUN_DIR/run.log"
