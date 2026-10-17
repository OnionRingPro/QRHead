#!/bin/bash
#SBATCH --job-name=qrhead-export-failures
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data
OUT_DIR=./output/multisession_failure_annotation

python exp_scripts/old_analysis/export_failures_for_annotation.py \
    --retrieval_data_file "$DATA/longmemeval_data/other_s.json" \
    --reference_file "$DATA/longmemeval_data/other_s_original.json" \
    --retrieval_result_file "./output/lme_topk_5/lme_retrieval.json" \
    --top5_generation_file "./output/lme_topk_5/lme_generation.json" \
    --top5_eval_results "./output/lme_topk_5/lme_generation.json.eval-results" \
    --oracle_generation_file "./output/lme_topk_5_oracle/lme_generation_oracle.json" \
    --oracle_eval_results "./output/lme_topk_5_oracle/lme_generation_oracle.json.eval-results" \
    --question_type multi-session \
    --k 5 \
    --out_txt "$OUT_DIR/multisession_failures_k5.txt" \
    --out_json "$OUT_DIR/multisession_failures_k5.json"
