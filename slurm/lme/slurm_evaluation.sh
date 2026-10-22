#!/bin/bash
#SBATCH --job-name=qrhead-generation-eval
#SBATCH --account=aip-xiye17
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out
#SBATCH --error=./out/%x_%j.err
#SBATCH --mail-user=chenqi3@ualberta.ca
#SBATCH --mail-type=BEGIN,END,FAIL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$SCRATCH/hf_cache

# arrow/24.0.0 injects pyarrow into the environment (it's a module, not a pip
# package). The vLLM OpenAI server lazy-imports outlines -> datasets -> pyarrow
# on every chat completion request regardless of whether guided decoding is
# used, so this must be loaded before the server starts or every request 500s.
module load StdEnv/2023 python/3.11.5 arrow/24.0.0

# Same vllm venv used for generation (main .venv has no vllm installed).
source "$SCRATCH/venvs/qrhead_vllm/bin/activate"

DATA=~/scratch/qrhead_data/data
OUT=./out
PORT=8000
# Local HF cache snapshot, same model used to generate lme_generation.json.
# Avoids any network/hub lookup on the compute node.
MODEL_PATH="$HOME/projects/aip-xiye17/chenqi3/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# README's two commands assume two separate terminals: serve_vllm.sh never
# returns (it's a long-running server), so it has to be backgrounded here and
# the eval script has to wait until the server is actually ready to answer.
bash exp_scripts/generation/serve_vllm.sh 0 "$MODEL_PATH" "$PORT" 1 &
SERVER_PID=$!

echo "Waiting for vLLM judge server to come up on port $PORT..."
for i in $(seq 1 120); do
    if curl -s -o /dev/null "http://localhost:$PORT/health"; then
        echo "Server ready after $((i * 10)) seconds."
        break
    fi
    sleep 10
done

python exp_scripts/generation/eval_generation_lme.py \
    --base_url "http://localhost:$PORT/v1" \
    --metric_model "$MODEL_PATH" \
    --generation_result_file "$OUT/lme_generation_oracle.json" \
    --reference_file "$DATA/longmemeval_data/other_s_original.json"
EVAL_EXIT_CODE=$?

kill "$SERVER_PID" 2>/dev/null
exit $EVAL_EXIT_CODE
