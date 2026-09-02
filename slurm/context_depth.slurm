#!/bin/bash
#SBATCH --job-name=c6-depth
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/imi/uqv56xi/drq/slurm_logs/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/imi/uqv56xi/drq/slurm_logs/%x-%j.err
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=imi@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread

# Usage:
#   sbatch slurm/context_depth.slurm full ROOT_INDEX
#   sbatch slurm/context_depth.slurm four ROOT_INDEX
#
# Jobs for different (mode, root) pairs are independent and may run in parallel.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch $0 full|four ROOT_INDEX" >&2
    exit 2
fi

MODE="$1"
ROOT_INDEX="$2"

case "$MODE" in
    full|four) ;;
    *) echo "MODE must be full or four" >&2; exit 2 ;;
esac

if ! [[ "$ROOT_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ROOT_INDEX must be a non-negative integer" >&2
    exit 2
fi

REPO_ROOT="/lustre/fsn1/projects/rech/imi/uqv56xi/drq"
RUNS_DIR="${C6_RUNS_DIR:-$REPO_ROOT/data}"
RUN_DIR="$RUNS_DIR/c6_root_$ROOT_INDEX"
HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"
ROUND="${C6_CONTEXT_ROUND:-4}"
TIMEOUT="${C6_CONTEXT_TIMEOUT:-7200}"

for filename in args.pkl all_rounds_map_elites.pkl c6_rerank.json; do
    if [[ ! -f "$RUN_DIR/$filename" ]]; then
        echo "Missing input: $RUN_DIR/$filename" >&2
        exit 1
    fi
done

mkdir -p "$REPO_ROOT/slurm_logs"

module purge
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drq

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ "$MODE" == "full" ]]; then
    OUTPUT="$RUN_DIR/c6_full_permutations.json"
else
    OUTPUT="$RUN_DIR/c6_four_player.json"
fi

cd "$REPO_ROOT/src"

echo "=== JOB $SLURM_JOB_ID mode=$MODE root=$ROOT_INDEX at $(date) ==="
echo "Run: $RUN_DIR"
echo "CPU workers: $SLURM_CPUS_PER_TASK"
echo "Output: $OUTPUT"

srun --ntasks=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --cpu-bind=cores \
     python -u c6_context_depth.py \
       --run-dir "$RUN_DIR" \
       --human-dir "$HUMAN_DIR" \
       --mode "$MODE" \
       --round "$ROUND" \
       --n-processes "$SLURM_CPUS_PER_TASK" \
       --timeout "$TIMEOUT" \
       --output "$OUTPUT"

echo "=== Completed at $(date) ==="
echo "Result: $OUTPUT"
