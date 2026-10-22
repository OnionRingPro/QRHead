#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# Evaluate several oolong retrieval results side by side. Submit with slurm/submit.sh; variables:
#   RUNS       space-separated retrieval result files (required)
#   DATA_FILE  default $DATA_ROOT/oolong_data/synth_validation.json
# Writes $RUN_DIR/eval.txt.

source slurm/common.sh
: "${RUNS:?}"
DATA_FILE=${DATA_FILE:-$DATA_ROOT/oolong_data/synth_validation.json}

: > "$RUN_DIR/eval.txt"
for run in $RUNS; do
    echo "==== $run" | tee -a "$RUN_DIR/eval.txt"
    python exp_scripts/retrieval/eval_retrieval_oolong.py \
        --retrieval_result_file "$run" \
        --data_file "$DATA_FILE" \
        --auroc | tee -a "$RUN_DIR/eval.txt"
    echo | tee -a "$RUN_DIR/eval.txt"
done
