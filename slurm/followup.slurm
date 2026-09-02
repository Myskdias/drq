#!/bin/bash
#SBATCH --job-name=c6-followup
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/imi/uqv56xi/drq/slurm_logs/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/imi/uqv56xi/drq/slurm_logs/%x-%j.err
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=imi@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

# Usage: sbatch slurm/followup.slurm fixed|mechanism|both ROOT_INDEX
# Create slurm_logs before submitting: Slurm opens its logs before this script.
# Submit at most one job per (mode, root); both must not overlap with either mode.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch $0 fixed|mechanism|both ROOT_INDEX" >&2
    exit 2
fi

MODE="$1"
ROOT_INDEX="$2"
case "$MODE" in
    fixed) MODE_ARGS=(--fixed-only) ;;
    mechanism) MODE_ARGS=(--mechanism-only) ;;
    both) MODE_ARGS=() ;;
    *) echo "Mode must be fixed, mechanism, or both" >&2; exit 2 ;;
esac
if ! [[ "$ROOT_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ROOT_INDEX must be a non-negative integer" >&2
    exit 2
fi

REPO_ROOT="/lustre/fsn1/projects/rech/imi/uqv56xi/drq"
RUNS_DIR="${C6_RUNS_DIR:-$REPO_ROOT/data}"
RUN_DIR="$RUNS_DIR/c6_root_$ROOT_INDEX"
HUMAN_DIR="${C6_HUMAN_DIR:-$REPO_ROOT/human_warriors}"
TIMEOUT="${C6_FOLLOWUP_TIMEOUT:-7200}"
MECHANISM_ROUND="${C6_MECHANISM_ROUND:-4}"

for filename in args.pkl all_rounds_map_elites.pkl c6_rerank.json; do
    if [[ ! -f "$RUN_DIR/$filename" ]]; then
        echo "Missing input: $RUN_DIR/$filename" >&2
        exit 1
    fi
done

module purge
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drq

# One thread per worker; each Slurm allocation gets its own process pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT/src"
echo "=== JOB $SLURM_JOB_ID mode=$MODE root=$ROOT_INDEX on $(hostname) at $(date) ==="
echo "Run: $RUN_DIR"
echo "CPU workers: $SLURM_CPUS_PER_TASK"

srun --ntasks=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --cpu-bind=cores \
     python -u c6_followup.py \
       --run-dir "$RUN_DIR" \
       --human-dir "$HUMAN_DIR" \
       --n-processes "$SLURM_CPUS_PER_TASK" \
       --timeout "$TIMEOUT" \
       --mechanism-round "$MECHANISM_ROUND" \
       --fixed-output "$RUN_DIR/c6_fixed_order.json" \
       --mechanism-output "$RUN_DIR/c6_mechanism.json" \
       "${MODE_ARGS[@]}"

echo "=== Follow-up completed at $(date) ==="
if [[ "$MODE" != mechanism ]]; then
    echo "Result: $RUN_DIR/c6_fixed_order.json"
fi
if [[ "$MODE" != fixed ]]; then
    echo "Result: $RUN_DIR/c6_mechanism.json"
fi
