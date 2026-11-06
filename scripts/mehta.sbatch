#!/bin/bash
# Parallel ion_npe command on the mehta node.
# 12 tasks, 48 GB, 8 hours. Rank 0 runs the command and trains.
# Every task simulates its own share of each collection batch.
#
#   sbatch scripts/mehta.sbatch nearby -o outputs/electrodes --noise 0 --seeds 0 \
#       -n 2000 --per-round 1000 --iterations 3 --epochs 40 --lbfgs 15
#SBATCH --job-name=ion-npe
#SBATCH --account=mehta
#SBATCH --partition=mehta
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -n 12
#SBATCH --output=/share/mehta/zx296/slurm_jobs/ion-npe-%j.out
#SBATCH --error=/share/mehta/zx296/slurm_jobs/ion-npe-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch scripts/mehta.sbatch <ion_npe arguments>" >&2
  exit 2
fi

cd /share/mehta/zx296/jc_transport_opt
mkdir -p /share/mehta/zx296/slurm_jobs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export ION_NPE_FARM="${PWD}/outputs/farm/${SLURM_JOB_ID}"
mkdir -p "${ION_NPE_FARM}"

echo "host $(hostname)  tasks 12  start $(date)"
echo "command: ion_npe $*"

# The batch shell only waits. --overlap lets the 12 tasks use this allocation.
srun --overlap --ntasks=12 --cpus-per-task=1 --cpu-bind=cores \
  "${PWD}/.venv/bin/python" -u -m ion_npe "$@"

echo "finished $(date)"
