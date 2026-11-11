#!/bin/bash
# Parallel ion_npe command on the mehta node.
# The mehta partition is one machine, mehta-cpu-01, with 16 cores.
# This allocation takes all of them. Rank 0 runs the command and trains.
# Every task simulates its own share of each collection batch.
#
#   sbatch scripts/mehta.sbatch nearby -o outputs/electrodes --noise 0 --seeds 0 \
#       -n 2000 --per-round 1000 --iterations 3 --epochs 40 --lbfgs 15
#SBATCH --job-name=ion-npe
#SBATCH --account=mehta
#SBATCH --partition=mehta
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
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
export ION_FARM_DIR="${PWD}/outputs/farm/${SLURM_JOB_ID}"
mkdir -p "${ION_NPE_FARM}" "${ION_FARM_DIR}"

echo "host $(hostname)  nodes ${SLURM_NNODES}  tasks ${SLURM_NTASKS}  start $(date)"
echo "command: ion_npe $*"

# The batch shell only waits. --overlap lets every allocated task use this step.
srun --overlap --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-${SLURM_NTASKS}}" \
  --cpus-per-task=1 --cpu-bind=cores \
  "${PWD}/.venv/bin/python" -u -m ion_npe "$@"

echo "finished $(date)"
