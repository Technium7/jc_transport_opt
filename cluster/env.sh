# Sourced by the Slurm scripts. Point ION_PYTHON at the trapsim environment
# on G2: it needs numpy, scipy, and the electrode package.
#
#   export ION_PYTHON=$HOME/.conda/envs/trapsim/bin/python
#
# The laptop path is used when that interpreter is present. Otherwise the
# scripts call `python` from the job environment.

if [[ -z "${ION_PYTHON:-}" ]]; then
  if [[ -x /opt/anaconda3/envs/trapsim/bin/python ]]; then
    ION_PYTHON=/opt/anaconda3/envs/trapsim/bin/python
  else
    ION_PYTHON=python
  fi
fi
export ION_PYTHON
