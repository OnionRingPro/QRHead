#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# Attention retrieval on oolong, then its eval. Submit with slurm/submit.sh; variables:
#   RETRIEVER     qr_head | cot_qr_head | full_head                                   (required)
#   MODEL_CONFIG  e.g. src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml (required)
#   DATA_FILE     default $DATA_ROOT/oolong_data/synth_validation.json
#   SUBSAMPLE, MAX_INSTANCES                    optional, for quick runs
#   MAX_NEW_TOKENS, Q_WEIGHT, COT_INSTRUCTION, COT_FILE, NULL_COT=1   optional, cot_qr_head only
#                                               (MAX_NEW_TOKENS=0 is the no-CoT control)
#   SPAN_FROM_QUESTION=1                        cot_qr_head only: score from the question, not the instruction
#                                               (the pre-2026-10-24 behaviour, ~7-15 points lower on oolong)
# Writes $RUN_DIR/retrieval.json (+ retrieval_cot.json for cot_qr_head) and $RUN_DIR/eval.txt
# (no eval when SUBSAMPLE/MAX_INSTANCES is set).

source slurm/common.sh
: "${RETRIEVER:?}" "${MODEL_CONFIG:?}"
DATA_FILE=${DATA_FILE:-$DATA_ROOT/oolong_data/synth_validation.json}

args=()
[ -n "${SUBSAMPLE:-}" ] && args+=(--subsample "$SUBSAMPLE")
[ -n "${MAX_INSTANCES:-}" ] && args+=(--max_instances "$MAX_INSTANCES")
[ -n "${MAX_NEW_TOKENS:-}" ] && args+=(--max_new_tokens "$MAX_NEW_TOKENS")
[ -n "${Q_WEIGHT:-}" ] && args+=(--q_weight "$Q_WEIGHT")
[ -n "${COT_INSTRUCTION:-}" ] && args+=(--cot_instruction "$COT_INSTRUCTION")
[ -n "${COT_FILE:-}" ] && args+=(--cot_file "$COT_FILE")
[ "${NULL_COT:-0}" = 1 ] && args+=(--null_cot)
[ "${SPAN_FROM_QUESTION:-0}" = 1 ] && args+=(--span_from_question)

python exp_scripts/retrieval/run_retrieval.py \
    --input_file "$DATA_FILE" \
    --output_file "$RUN_DIR/retrieval.json" \
    --data_type oolong \
    --retriever_type "$RETRIEVER" \
    --config_or_config_path "$MODEL_CONFIG" \
    "${args[@]}"

# the eval scores every question in DATA_FILE, so it only fits a run over all of them
if [ -n "${SUBSAMPLE:-}${MAX_INSTANCES:-}" ]; then
    echo "skipping eval: SUBSAMPLE/MAX_INSTANCES ran only part of $DATA_FILE"
else
    python exp_scripts/retrieval/eval_retrieval_oolong.py \
        --retrieval_result_file "$RUN_DIR/retrieval.json" \
        --data_file "$DATA_FILE" \
        --auroc | tee "$RUN_DIR/eval.txt"
fi
