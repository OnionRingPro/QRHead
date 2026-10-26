#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# Score oolong generations side by side (oolong-synth rules, as in the QR-CoT repo). CPU only. Submit with
# slurm/submit.sh; variables:
#   RUNS       space-separated generate.sh run dirs and/or generation .jsonl files (required)
#   DATA_FILE  default $DATA_ROOT/oolong_data/synth_test_64k_all.json; must be the file the generations were made from
# Writes $RUN_DIR/eval.txt, and scores_<name>.jsonl next to every generation file it scores.
# generate.sh already scores its own outputs when it finishes; this is for rescoring and for comparing runs.

source slurm/common.sh
deactivate # the scorer needs python-dateutil, which the vLLM venv has and .venv does not
source slurm/vllm_env.sh
source "$VLLM_ENV/bin/activate"
: "${RUNS:?}"
DATA_FILE=${DATA_FILE:-$DATA_ROOT/oolong_data/synth_test_64k_all.json}

python exp_scripts/generation/eval_generation_oolong.py \
    --data_file "$DATA_FILE" \
    --runs $RUNS | tee "$RUN_DIR/eval.txt" # $RUNS unquoted: one argument per run
