#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# vLLM answers on oolong over all / random / retrieved lines (QR-CoT's generation step). Submit with slurm/submit.sh; variables:
#   METHOD          full | random | retrieval                                        (required)
#   TOPK            k's for random/retrieval, e.g. "20 50 100 200 400"
#   RETRIEVAL_FILE  retrieval only: a run_retrieval.py retrieval.json (e.g. a qrcot run on this run's full.jsonl)
#   DATA_FILE       default $DATA_ROOT/oolong_data/synth_test_64k_all.json
#   MODEL           default Qwen/Qwen3-4B-Thinking-2507
#   MAX_INSTANCES   optional, the first N questions (pilots)
#   MAX_TOKENS, MAX_MODEL_LEN   optional, defaults 16384 and 262144
# Writes $RUN_DIR/full.jsonl or $RUN_DIR/<method>_<k>.jsonl, appending every few questions, then scores them into
# $RUN_DIR/eval.txt (+ scores_<name>.jsonl). The 2:59 limit gets the shortest queue; if it runs out, resubmit with
# RESUME=1 and the same tag to continue where it stopped.
# Two GPUs because the 262144 window's KV cache (~36 GiB for one full sequence of Qwen3-4B) does not fit next to the
# weights on one L40S.

source slurm/common.sh
deactivate # leave .venv: vLLM needs its own torch / transformers
source slurm/vllm_env.sh
source "$VLLM_ENV/bin/activate"
unset PYTORCH_CUDA_ALLOC_CONF # common.sh's expandable_segments is for the HF retrievers; vLLM manages its own memory
export HF_HUB_OFFLINE=1 # the models are already in $HF_HOME; don't reach for the hub from the compute node
# compile caches on the node's local disk: shared on $SCRATCH, two jobs on different nodes writing and reading the same
# Triton kernel file failed one of them with "Stale file handle" (2026-10-25). Compiling anew costs ~1 min per job.
export VLLM_CACHE_ROOT=$SLURM_TMPDIR/vllm_cache
export TRITON_CACHE_DIR=$SLURM_TMPDIR/triton_cache

: "${METHOD:?}"
DATA_FILE=${DATA_FILE:-$DATA_ROOT/oolong_data/synth_test_64k_all.json}

args=()
[ -n "${TOPK:-}" ] && args+=(--topk $TOPK) # unquoted: one argument per k
[ -n "${RETRIEVAL_FILE:-}" ] && args+=(--retrieval_file "$RETRIEVAL_FILE")
[ -n "${MODEL:-}" ] && args+=(--model "$MODEL")
[ -n "${MAX_INSTANCES:-}" ] && args+=(--max_instances "$MAX_INSTANCES")
[ -n "${MAX_TOKENS:-}" ] && args+=(--max_tokens "$MAX_TOKENS")
[ -n "${MAX_MODEL_LEN:-}" ] && args+=(--max_model_len "$MAX_MODEL_LEN")

python exp_scripts/generation/run_generation_oolong.py \
    --input_file "$DATA_FILE" \
    --output_dir "$RUN_DIR" \
    --method "$METHOD" \
    "${args[@]}" || exit 1

python exp_scripts/generation/eval_generation_oolong.py \
    --data_file "$DATA_FILE" \
    --runs "$RUN_DIR" | tee "$RUN_DIR/eval.txt"
