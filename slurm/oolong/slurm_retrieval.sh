#!/bin/bash
#SBATCH --job-name=oolong_retrieval
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/oolong_retrieval_%x_%j.out
#SBATCH --error=./out/oolong_retrieval_%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=$SCRATCH/qrhead_data/data/oolong_data
OUT=./output/oolong_cot
mkdir -p $OUT

python exp_scripts/retrieval/run_retrieval.py \
    --input_file "$DATA/synth_validation.json" \
    --output_file "$OUT/synth_validation_cot_qr_control.json" \
    --data_type oolong \
    --retriever_type cot_qr_head \
    --max_new_tokens 0 \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml



