#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# Attention retrieval on oolong, then its eval. Submit with slurm/submit.sh; variables:
#   RETRIEVER     qr_head | cot_qr_head | qrcot | full_head                           (required)
#   MODEL_CONFIG  e.g. src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml (required)
#   DATA_FILE     default $DATA_ROOT/oolong_data/synth_validation.json
#   SUBSAMPLE, MAX_INSTANCES                    optional, for quick runs
#   MAX_NEW_TOKENS, COT_INSTRUCTION, COT_FILE   optional, cot_qr_head only (MAX_NEW_TOKENS=0 is the no-CoT control)
#   GEN_FILE      qrcot only (required): JSONL of full-context generations ('idx', 'formatted_input', 'output')
# Writes $RUN_DIR/retrieval.json (+ retrieval_cot.json for cot_qr_head) and $RUN_DIR/eval.txt
# (no eval when SUBSAMPLE/MAX_INSTANCES is set).

source slurm/common.sh
: "${RETRIEVER:?}" "${MODEL_CONFIG:?}"
DATA_FILE=${DATA_FILE:-$DATA_ROOT/oolong_data/synth_validation.json}

args=()
[ -n "${SUBSAMPLE:-}" ] && args+=(--subsample "$SUBSAMPLE")
[ -n "${MAX_INSTANCES:-}" ] && args+=(--max_instances "$MAX_INSTANCES")
[ -n "${MAX_NEW_TOKENS:-}" ] && args+=(--max_new_tokens "$MAX_NEW_TOKENS")
[ -n "${COT_INSTRUCTION:-}" ] && args+=(--cot_instruction "$COT_INSTRUCTION")
[ -n "${COT_FILE:-}" ] && args+=(--cot_file "$COT_FILE")
[ -n "${GEN_FILE:-}" ] && args+=(--generation_file "$GEN_FILE")

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
