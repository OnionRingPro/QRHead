#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# Build the vLLM venv that slurm/oolong/generate.sh uses. Submit with slurm/submit.sh; variables:
#   VLLM_VERSION  default 0.32.0
#   VLLM_ENV      default $SCRATCH/venvs/vllm; refused if it already exists
# $SCRATCH is purged when idle, so $RUN_DIR/freeze.txt keeps what to reinstall.
set -euo pipefail
: "${RUN_DIR:?submit this job with slurm/submit.sh, which sets RUN_DIR}"
source "$RUN_DIR/params.env"
source slurm/vllm_env.sh
VLLM_VERSION=${VLLM_VERSION:-0.32.0}
if [ -e "$VLLM_ENV" ]; then
    echo "$VLLM_ENV exists; remove it first or set VLLM_ENV" >&2
    exit 1
fi

virtualenv --no-download "$VLLM_ENV"
source "$VLLM_ENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index "vllm==$VLLM_VERSION" python-dateutil
pip freeze > "$RUN_DIR/freeze.txt"
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
