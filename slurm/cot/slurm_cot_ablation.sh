#!/bin/bash
#SBATCH --job-name=qrhead-cot-ablation
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%A_%a.out
#SBATCH --error=./out/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chenqi3@ualberta.ca

# CoT instruction ablation on a fixed random subset of LME (seed 42 in run_retrieval.py), one array task per instruction.
# Compare afterwards with slurm/cot/slurm_cot_compare.sh.
# No-CoT control (same instructions, query = question only):
#   sbatch --time=01:00:00 --export=ALL,MAX_NEW_TOKENS=0,OUT_SUFFIX=_nocot slurm/cot/slurm_cot_ablation.sh
NAMES=(current A_con B_quote C_con_escape)
INSTRUCTIONS=(
    "Think step by step about the following query using the paragraphs above, then answer it."
    "Answer the following query step by step: first extract all the relevant information from the paragraphs above, and then reason over the information to get the answer."
    "First quote the parts of the paragraphs above that are relevant to the following query, then answer it."
    "Answer the following query step by step: first extract all the relevant information from the paragraphs above, and then reason over the information to get the answer. If no relevant information is found, just say so briefly."
)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
SUBSAMPLE=${SUBSAMPLE:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data
OUT_DIR=./output/lme_cot_ablation/$NAME${OUT_SUFFIX:-}
mkdir -p "$OUT_DIR"

python exp_scripts/retrieval/run_retrieval.py \
    --input_file "$DATA/longmemeval_data/other_s.json" \
    --output_file "$OUT_DIR/lme_cot_retrieval.json" \
    --data_type lme \
    --retriever_type cot_qr_head \
    --cot_instruction "${INSTRUCTIONS[$SLURM_ARRAY_TASK_ID]}" \
    --subsample "$SUBSAMPLE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
