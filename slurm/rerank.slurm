#!/bin/bash
#SBATCH --job-name=c6-rerank-root2
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

set -euo pipefail
set -x

echo "=== JOB $SLURM_JOB_ID start on $(hostname) at $(date) ==="

REPO_ROOT="/lustre/fsn1/projects/rech/imi/uqv56xi/drq"
RUN_DIR="$REPO_ROOT/data/c6_root_2"
HUMAN_DIR="$REPO_ROOT/human_warriors"

module purge
module load miniforge/24.9.0
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate drq

# Un thread par worker ; les simulations utilisent 24 processus.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Vérifier que les données du root sont présentes.
for filename in args.pkl all_rounds_map_elites.pkl; do
    if [[ ! -f "$RUN_DIR/$filename" ]]; then
        echo "Fichier manquant : $RUN_DIR/$filename" >&2
        exit 1
    fi
done

cd "$REPO_ROOT/src"

echo "Run : $RUN_DIR"
echo "Workers CPU : $SLURM_CPUS_PER_TASK"

srun --ntasks=1 \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --cpu-bind=cores \
     python -u c6_rerank.py \
       --run-dir "$RUN_DIR" \
       --human-dir "$HUMAN_DIR" \
       --n-processes "$SLURM_CPUS_PER_TASK" \
       --repeats 24 \
       --timeout 21600 \
       --min-candidates 8 \
       --max-candidates 12 \
       --output "$RUN_DIR/c6_rerank.json"

echo "=== Reranking terminé à $(date) ==="
echo "Résultat : $RUN_DIR/c6_rerank.json"