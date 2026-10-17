#!/bin/bash
#SBATCH --job-name=qrhead-cot-retrieval
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chenqi3@ualberta.ca

# Smoke test (first 2 queries, separate output dir):
#   sbatch --time=01:00:00 --export=ALL,MAX_INSTANCES=2,OUT_DIR=./output/lme_cot_smoke slurm/cot/slurm_cot_retrieval.sh
# Full run (resumes from the existing output file if the job gets cut off):
#   sbatch slurm/cot/slurm_cot_retrieval.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate

DATA=~/scratch/qrhead_data/data
OUT_DIR=${OUT_DIR:-./output/lme_cot}
MAX_INSTANCES=${MAX_INSTANCES:-0}
mkdir -p "$OUT_DIR"

# writes $OUT_DIR/lme_cot_retrieval.json (doc scores) and $OUT_DIR/lme_cot_retrieval_cot.json (CoT text + ids)
python exp_scripts/retrieval/run_retrieval.py \
    --input_file "$DATA/longmemeval_data/other_s.json" \
    --output_file "$OUT_DIR/lme_cot_retrieval.json" \
    --data_type lme \
    --retriever_type cot_qr_head \
    --max_instances "$MAX_INSTANCES" \
    --config_or_config_path src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
