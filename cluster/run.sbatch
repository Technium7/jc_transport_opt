#!/bin/bash
#SBATCH --job-name=ion-npe
#SBATCH --partition=mehta
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Slurm --mem is per node, same as:
#   srun --mem=4G --time=8:00:00 -n 1 -p mehta --pty /bin/bash
# 10 nodes x 4G = 40G. The account cap is 16 nodes and 128G total, so
# 16 nodes x 8G is the ceiling:
#   sbatch --nodes=16 --ntasks=16 --mem=8G cluster/run.sbatch joint -o outputs/joint
# Do not request a total (nodes times --mem) above 128G.
#
# Submit from the project directory, after `mkdir -p logs`:
#   sbatch cluster/run.sbatch collect -n 400 --seeds 0 1 2 3 -o data/train.npz
#   sbatch --nodes=1 --ntasks=1 cluster/run.sbatch train -d data/train.npz -o data/models
#   sbatch cluster/run.sbatch optimize -d data/train.npz -m data/models/stochastic.npz
#   sbatch cluster/run.sbatch study -o outputs/study
#   sbatch cluster/run.sbatch joint -o outputs/joint
#   sbatch cluster/run.sbatch nearby -o outputs/nearby
#
# `train` uses one task. Collection and the search use every task in the allocation.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs data outputs
if [[ -f cluster/env.sh ]]; then
  # shellcheck disable=SC1091
  source cluster/env.sh
fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY="${ION_PYTHON:-python}"

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch cluster/run.sbatch <ion_npe arguments>" >&2
  exit 1
fi

echo "job ${SLURM_JOB_ID} on ${SLURM_NNODES} nodes, ${SLURM_NTASKS} tasks, python ${PY}"
echo "command: $*"

if [[ "$1" == "train" ]]; then
  exec srun --ntasks=1 --nodes=1 --export=ALL --chdir="${SLURM_SUBMIT_DIR}" \
    "$PY" -m ion_npe "$@"
fi

exec srun --ntasks="${SLURM_NTASKS}" --nodes="${SLURM_NNODES}" --export=ALL \
  --chdir="${SLURM_SUBMIT_DIR}" "$PY" -m ion_npe "$@"
