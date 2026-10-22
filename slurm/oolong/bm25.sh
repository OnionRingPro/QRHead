#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:59:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# BM25 baselines on oolong, each followed by its eval. Submit with slurm/submit.sh; variables:
#   COT_FILE   retrieval_cot.json of a cot_qr_head run; needed for the cot and q+cot modes
#   MODES      default "q", or "q cot q+cot" when COT_FILE is set
#   DATA_FILE  default $DATA_ROOT/oolong_data/synth_validation.json
# Writes $RUN_DIR/bm25_<mode>.json and $RUN_DIR/eval.txt.

source slurm/common.sh
DATA_FILE=${DATA_FILE:-$DATA_ROOT/oolong_data/synth_validation.json}
if [ -n "${COT_FILE:-}" ]; then
    MODES=${MODES:-"q cot q+cot"}
    cot_args=(--cot_file "$COT_FILE")
else
    MODES=${MODES:-q}
    cot_args=()
fi

: > "$RUN_DIR/eval.txt"
for mode in $MODES; do
    python exp_scripts/retrieval/run_bm25.py \
        --input_file "$DATA_FILE" \
        --output_file "$RUN_DIR/bm25_$mode.json" \
        --mode "$mode" \
        "${cot_args[@]}"
    echo "==== bm25 $mode" | tee -a "$RUN_DIR/eval.txt"
    python exp_scripts/retrieval/eval_retrieval_oolong.py \
        --retrieval_result_file "$RUN_DIR/bm25_$mode.json" \
        --data_file "$DATA_FILE" \
        --auroc | tee -a "$RUN_DIR/eval.txt"
done
