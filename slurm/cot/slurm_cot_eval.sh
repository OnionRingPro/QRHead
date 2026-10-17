#!/bin/bash
#SBATCH --job-name=qrhead-cot-eval
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

# Retrieval recall of the full CoT run (slurm/cot/slurm_cot_retrieval.sh), and a paired comparison with QR.

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data

python exp_scripts/retrieval/eval_retrieval.py \
    --retrieval_result_file ./output/lme_cot/lme_cot_retrieval.json \
    --data_file "$DATA/longmemeval_data/other_s.json"

python exp_scripts/retrieval/compare_retrieval.py \
    --data_file "$DATA/longmemeval_data/other_s.json" \
    --k 5 \
    --runs qr=./output/lme_topk_5/lme_retrieval.json cot=./output/lme_cot/lme_cot_retrieval.json
