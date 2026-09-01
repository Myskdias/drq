#!/bin/bash
#SBATCH --job-name=drqsmoke
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/imi/uqv56xi/drq/slurm_logs/drqsmoke-%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/imi/uqv56xi/drq/slurm_logs/drqsmoke-%A_%a.err
#SBATCH --qos=qos_gpu_h100-dev     # dev / t3 / t4
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --account=imi@h100
#SBATCH --array=1

set -euo pipefail
set -x   # trace every command to the .err log so a fast failure still leaves a trail
echo "=== JOB $SLURM_JOB_ID start on $(hostname) at $(date) ==="


# ------------------------------------------------------------------
REPO_PATH="/lustre/fsn1/projects/rech/imi/uqv56xi/drq"
MODEL_PATH="/lustre/fsn1/projects/rech/imi/uqv56xi/models/Qwen/Qwen3-Coder-30B-A3B-Instruct"
# ------------------------------------------------------------------

# Conda
#source /lustre/fsn1/projects/rech/imi/uqv56xi/conda/etc/profile.d/conda.sh
module purge
module load arch/h100
module load python/3.12.2
module load cuda/12.8.0

conda activate drq

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_PATH/src"

INIT=$(python - <<'PY'
import json
with open("../c6_split.json") as f:
    print(json.load(f)["initialization"][0])
PY
)

echo "$INIT"

python drq.py \
  --seed 0 \
  --llm-seed 0 \
  --llm-backend huggingface \
  --gpt-model /lustre/fsn1/projects/rech/imi/uqv56xi/models/Qwen/Qwen3-Coder-30B-A3B-Instruct/ \
  --save-dir ../runs/smoke_root_0 \
  --initial-opps "../human_warriors/$INIT" \
  --n-rounds 4 \
  --n-iters 8 \
  --n-init 8 \
  --n-mutate 1 \
  --last-k-opps 20 \
  --warmup-with-init-opps True \
  --warmup-with-past-champs True \
  --fitness-threshold 999 \
  --n-processes 24 \
  --simargs.rounds 24