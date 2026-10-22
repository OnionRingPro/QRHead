#!/bin/bash
#SBATCH --job-name=qrhead-eval-by-type
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data
OUT=./out

python exp_scripts/retrieval/eval_retrieval_by_type.py \
    --retrieval_result_file "$OUT/lme_retrieval.json" \
    --data_file "$DATA/longmemeval_data/other_s.json" \
    --reference_file "$DATA/longmemeval_data/other_s_original.json"
