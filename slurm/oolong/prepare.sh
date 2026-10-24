#!/bin/bash
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead

# Convert oolong-synth parquet files into QRHead retrieval JSON. Submit with slurm/submit.sh; variables:
#   SPLIT            validation | test, default validation
#   MAX_CONTEXT_LEN  default: prepare_oolong.py's (131072)
#   MIN_CONTEXT_LEN  default: prepare_oolong.py's (0)
#   OUT_FILE         default $DATA_ROOT/oolong_data/synth_<SPLIT>.json; an existing file is kept unless OVERWRITE=1,
#                    since every retrieval run on it would silently change meaning
# The log in $RUN_DIR keeps the Stats line (how many questions each filter dropped).

module load StdEnv/2023 python/3.11.5 arrow/22.0.0 # pyarrow
source slurm/common.sh
SPLIT=${SPLIT:-validation}
OUT_FILE=${OUT_FILE:-$DATA_ROOT/oolong_data/synth_$SPLIT.json}
if [ -e "$OUT_FILE" ] && [ "${OVERWRITE:-0}" != 1 ]; then
    echo "$OUT_FILE exists; set OUT_FILE to a new path, or OVERWRITE=1" >&2
    exit 1
fi

args=()
[ -n "${MAX_CONTEXT_LEN:-}" ] && args+=(--max_context_len "$MAX_CONTEXT_LEN")
[ -n "${MIN_CONTEXT_LEN:-}" ] && args+=(--min_context_len "$MIN_CONTEXT_LEN")

python exp_scripts/data/prepare_oolong.py \
    --raw_dir "$SCRATCH/qrhead_data/raw/oolong-synth" \
    --output_file "$OUT_FILE" \
    --split "$SPLIT" \
    "${args[@]}"
