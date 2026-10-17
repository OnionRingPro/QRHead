#!/bin/bash
#SBATCH --job-name=qrhead-generation
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:10:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err
#SBATCH --mail-user=chenqi3@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$SCRATCH/hf_cache

source "$SCRATCH/venvs/qrhead_vllm/bin/activate"

DATA=~/scratch/qrhead_data/data
OUT=./out

# python exp_scripts/generation/run_generation_lme.py \
#     --input_file "$DATA/longmemeval_data/other_s_original.json" \
#     --output_file "$OUT/lme_generation_topk10.json" \
#     --model_path "$HOME/projects/aip-xiye17/chenqi3/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
#     --retrieval_method retriever \
#     --retrieval_result_file "$OUT/lme_retrieval.json" \
#     --topk_doc 10

python exp_scripts/generation/run_generation_lme.py \
    --input_file "$DATA/longmemeval_data/other_s_original.json" \
    --output_file "$OUT/lme_generation_oracle.json" \
    --model_path "$HOME/projects/aip-xiye17/chenqi3/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --retrieval_method oracle \
    --retrieval_result_file none \
    --topk_doc 5