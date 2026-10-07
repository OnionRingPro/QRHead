#!/bin/bash
#SBATCH --job-name=qrhead-crosstab
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data

python exp_scripts/retrieval/crosstab_multisession.py \
    --retrieval_result_file "./output/lme_topk_5/lme_retrieval.json" \
    --data_file "$DATA/longmemeval_data/other_s.json" \
    --reference_file "$DATA/longmemeval_data/other_s_original.json" \
    --top5_eval_results "./output/lme_topk_5/lme_generation.json.eval-results" \
    --oracle_eval_results "./output/lme_topk_5_oracle/lme_generation_oracle.json.eval-results" \
    --question_type multi-session \
    --k 5
