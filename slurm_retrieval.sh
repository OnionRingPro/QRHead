#!/bin/bash
#SBATCH --job-name=qrhead
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/qrhead_%x_%j.out
#SBATCH --error=./out/qrhead_%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data
OUT=./out
mkdir -p "$OUT"

# # retrieval on LME
# python exp_scripts/retrieval/run_retrieval.py \
#     --input_file "$DATA/longmemeval_data/other_s.json" \
#     --output_file "$OUT/lme_retrieval.json" \
#     --data_type lme \
#     --retriever_type qr_head \
#     --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml

# evaluate retrieval on LME
python exp_scripts/eval_retrieval.py \
    --retrieval_result_file "$OUT/lme_retrieval.json" \
    --data_file "$DATA/longmemeval_data/other_s.json"


