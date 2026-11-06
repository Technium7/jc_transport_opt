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

# One allocation for collection, training, and cross-entropy search.
# 10 nodes x 4G = 40G. Cap: 16 nodes and 128G total (16 x 8G).
# Slurm --mem is per node. Override with sbatch --nodes / --ntasks / --mem / --time.
#
#   mkdir -p logs
#   sbatch cluster/pipeline.sbatch
#
# Counts below match a cluster-sized run of the local collect / train / optimize
# commands. study, joint, and nearby already combine the three stages; submit
# those with cluster/run.sbatch instead of this pipeline.

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

N_COLLECT="${N_COLLECT:-400}"
EPOCHS="${EPOCHS:-200}"
LBFGS="${LBFGS:-40}"
ITERS="${ITERS:-4}"

echo "job ${SLURM_JOB_ID}: collect ${N_COLLECT}, train, optimize on ${SLURM_NNODES} nodes"

srun --ntasks="${SLURM_NTASKS}" --nodes="${SLURM_NNODES}" --export=ALL \
  --chdir="${SLURM_SUBMIT_DIR}" \
  "$PY" -m ion_npe collect -n "${N_COLLECT}" --seeds 0 1 2 3 --fraction 0.5 --seed 0 \
  -o data/train.npz

srun --ntasks=1 --nodes=1 --export=ALL --chdir="${SLURM_SUBMIT_DIR}" \
  "$PY" -m ion_npe train -d data/train.npz -o data/models \
  --epochs "${EPOCHS}" --lbfgs "${LBFGS}"

srun --ntasks="${SLURM_NTASKS}" --nodes="${SLURM_NNODES}" --export=ALL \
  --chdir="${SLURM_SUBMIT_DIR}" \
  "$PY" -m ion_npe optimize -d data/train.npz -m data/models/stochastic.npz \
  -o outputs/optimize --iterations "${ITERS}" --epochs "${EPOCHS}" --lbfgs "${LBFGS}"
