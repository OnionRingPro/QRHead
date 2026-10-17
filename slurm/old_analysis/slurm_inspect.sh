#!/bin/bash
#SBATCH --job-name=qrhead-inspect
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data

python exp_scripts/old_analysis/inspect_sample.py \
    --idx 3a704032 28dc39ac bf659f65 85fa3a3f gpt4_a56e767c \
    --retrieval_data_file "$DATA/longmemeval_data/other_s.json" \
    --reference_file "$DATA/longmemeval_data/other_s_original.json" \
    --retrieval_result_file "./output/lme_topk_5/lme_retrieval.json" \
    --generation_result_file "./output/lme_topk_5/lme_generation.json" \
    --eval_results_file "./output/lme_topk_5/lme_generation.json.eval-results"
