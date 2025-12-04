#!/bin/bash
# Bootstrap + activate the hm_env virtual environment with the exact steps
# required by our Slurm jobs (see run_all_v2_zero_shot.sbatch).

set -euo pipefail

# Mirror the scheduler scripts: reset modules, load python/pytorch, and clear
# site-packages that might leak into the venv.
module purge
module load python/3.10.4
module load pytorch/2.0.1
unset PYTHONPATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${HM_ENV_DIR:-$HOME/hm_env}"

if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Creating virtual environment at ${ENV_DIR}"
    python -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"

echo "Environment ready. To reuse later run:"
echo "  module purge && module load python/3.10.4 pytorch/2.0.1 && unset PYTHONPATH"
echo "  source ${ENV_DIR}/bin/activate"

