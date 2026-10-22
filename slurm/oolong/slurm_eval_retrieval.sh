#!/bin/bash
#SBATCH --job-name=oolong-eval
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

# Recall@|gold| (+ random baseline and AUROC) of every oolong retrieval run, broken down by filter_type.

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=$SCRATCH/qrhead_data/data/oolong_data

for run in \
    ./output/oolong_cot/synth_validation_qr.json \
    ./output/oolong_cot/synth_validation_cot_qr_control.json \
    ./output/oolong_cot/synth_validation_cot.json \
    ./output/oolong_bm25/synth_validation_bm25_q.json \
    ./output/oolong_bm25/synth_validation_bm25_cot.json \
    ./output/oolong_bm25/synth_validation_bm25_q+cot.json; do
    echo "==== $run"
    python exp_scripts/retrieval/eval_retrieval_oolong.py \
        --retrieval_result_file "$run" \
        --data_file "$DATA/synth_validation.json" \
        --auroc
    echo
done
