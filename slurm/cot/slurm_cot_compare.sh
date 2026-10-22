#!/bin/bash
#SBATCH --job-name=qrhead-cot-compare
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

# Recall@5 of each CoT instruction vs the original QR run, on the queries all runs cover.
# Usually submitted with --dependency=afterok:<slurm/cot/slurm_cot_ablation.sh job id>.

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data
ABL=./output/lme_cot_ablation

# python exp_scripts/retrieval/fuse_retrieval.py \
#     --result_files ./output/lme_topk_5/lme_retrieval.json $ABL/current/lme_cot_retrieval.json \
#     --output_file $ABL/qr_current_rrf.json

python exp_scripts/retrieval/compare_retrieval.py \
    --data_file "$DATA/longmemeval_data/other_s.json" \
    --k 5 \
    --runs \
    qr=./output/lme_topk_5/lme_retrieval.json \
    cot=./output/lme_cot/lme_cot_retrieval.json \
    qw0.8=./output/lme_cot/lme_cot_qw0.8_retrieval.json




# --runs \
    #     qr=./output/lme_topk_5/lme_retrieval.json \
    #     current=$ABL/current/lme_cot_retrieval.json \
    #     cur_nocot=$ABL/current_nocot/lme_cot_retrieval.json \
    #     qr+current=$ABL/qr_current_rrf.json

# 50-query instruction ablation also had: A_con, B_quote, C_con_escape and their *_nocot runs under $ABL

