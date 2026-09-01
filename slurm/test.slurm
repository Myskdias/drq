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

echo
echo "===== QWEN / REDCODE SMOKE TEST ====="

export MODEL_PATH

python - <<'PY'
import asyncio
import os

from corewar_util import SimulationArgs, simargs_to_environment
from llm_corewar import CorewarGPT


model_path = os.environ["MODEL_PATH"]

with open("./prompts/system_prompt_0.txt") as f:
    system_prompt = f.read()

with open("./prompts/new_prompt_0.txt") as f:
    new_prompt = f.read()

with open("./prompts/mutate_prompt_0.txt") as f:
    mutate_prompt = f.read()

gpt = CorewarGPT(
    model=model_path,
    system_prompt=system_prompt,
    new_warrior_prompt=new_prompt,
    mutate_warrior_prompt=mutate_prompt,
    backend="huggingface",
    temperature=1.0,
    seed=0,
    max_new_tokens=1024,
    environment=simargs_to_environment(SimulationArgs()),
)

echo "Start gen at $(date) ==="
warriors = asyncio.run(
    gpt.new_warrior_async(
        n_warriors=1,
        n_responses=8,
    )
).flatten()

echo "End gen at $(date) ==="

valid = 0
valid_hashes = set()

for i, warrior in enumerate(warriors):
    print()
    print("=" * 80)

    if warrior.warrior is not None:
        valid += 1
        valid_hashes.add(warrior.id)

        print(f"{i}: VALID - {warrior.warrior.name}")
        print(f"SHA256: {warrior.id}")
        print("RAW CODE:")
        print(warrior.llm_response)
    else:
        print(f"{i}: INVALID - {warrior.error}")
        print(f"SHA256: {warrior.id}")
        print("RAW RESPONSE:")
        print(warrior.llm_response)

print()
print(f"VALIDITY: {valid}/8 = {valid / 8:.1%}")
print(f"DISTINCT VALID HASHES: {len(valid_hashes)}/{valid}")
PY