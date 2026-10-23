# Shared setup for every job script submitted with slurm/submit.sh.
# Source it right after the #SBATCH header; jobs start in the repo root (#SBATCH --chdir).

QRHEAD=/project/aip-xiye17/chenqi3/QRHead
DATA_ROOT=$SCRATCH/qrhead_data/data

export HF_HOME=/project/aip-xiye17/chenqi3/hf_cache # cached models (same as ~/.bashrc)
# ~/.bashrc also sets the deprecated TRANSFORMERS_CACHE to the cache root, but `hf download` writes to
# $HF_HOME/hub, so transformers would miss newly downloaded models (e.g. Qwen3). Llama-3.1-8B is in both
# places with identical blobs, so it loads the same weights either way.
unset TRANSFORMERS_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $QRHEAD/.venv/bin/activate

# submit.sh exports RUN_DIR and writes the job's variables to params.env there
: "${RUN_DIR:?submit this job with slurm/submit.sh, which sets RUN_DIR}"
source "$RUN_DIR/params.env"

echo "job $SLURM_JOB_ID on $(hostname) | run dir $RUN_DIR | commit $(git -C $QRHEAD rev-parse --short HEAD)"
