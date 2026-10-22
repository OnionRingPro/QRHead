#!/bin/bash
#SBATCH --job-name=oolong-bm25
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:59:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=$SCRATCH/qrhead_data/data/oolong_data
OUT=./output/oolong_bm25
mkdir -p $OUT

for mode in q cot q+cot; do
    python exp_scripts/retrieval/run_bm25.py \
        --input_file "$DATA/synth_validation.json" \
        --output_file "$OUT/synth_validation_bm25_$mode.json" \
        --mode "$mode" \
        --cot_file "./out/oolong/synth_validation_cot_cot.json" \
        --q_repeat 5
done



