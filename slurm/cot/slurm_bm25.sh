#!/bin/bash
#SBATCH --job-name=qrhead-bm25
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err

# BM25 baselines (question / CoT / question+CoT) on LME, compared with QR and the attention-based CoT runs.

source ~/projects/aip-xiye17/chenqi3/QRHead/.venv/bin/activate  # needs rank_bm25 installed in this venv

DATA=~/scratch/qrhead_data/data
COT=./output/lme_cot/lme_cot_retrieval_cot.json
OUT=./output/lme_bm25
mkdir -p "$OUT"

for mode in q cot q+cot; do
    python exp_scripts/retrieval/run_bm25.py \
        --input_file "$DATA/longmemeval_data/other_s.json" \
        --output_file "$OUT/bm25_$mode.json" \
        --mode "$mode" \
        --cot_file "$COT"
done

# hybrid lexical + attention: RRF of QR with BM25 (the question-only one is the control for the CoT one)
for mode in q q+cot; do
    python exp_scripts/retrieval/fuse_retrieval.py \
        --result_files ./output/lme_topk_5/lme_retrieval.json "$OUT/bm25_$mode.json" \
        --output_file "$OUT/rrf_qr_bm25_$mode.json"
done

python exp_scripts/retrieval/compare_retrieval.py \
    --data_file "$DATA/longmemeval_data/other_s.json" \
    --k 5 \
    --runs \
        qr=./output/lme_topk_5/lme_retrieval.json \
        bm25_q=$OUT/bm25_q.json \
        bm25_cot=$OUT/bm25_cot.json \
        bm25_q+cot=$OUT/bm25_q+cot.json \
        cot=./output/lme_cot/lme_cot_retrieval.json \
        qw0.8=./output/lme_cot/lme_cot_qw0.8_retrieval.json \
        qr+bm25_q=$OUT/rrf_qr_bm25_q.json \
        qr+bm25_q+cot=$OUT/rrf_qr_bm25_q+cot.json
