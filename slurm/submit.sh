#!/bin/bash
# Submit a job script into its own run directory, output/<exp>/<date>_<tag>/  (run on the login node)
#
#   slurm/submit.sh <exp> <tag> <job script> [VAR=value ...] [-- extra sbatch options]
#   e.g. slurm/submit.sh oolong llama_qr slurm/oolong/retrieval.sh RETRIEVER=qr_head MODEL_CONFIG=src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
#
# The run directory gets: job.sh (a copy of the script, which is what Slurm runs, so later edits to the
# original never change this run), params.env (the VAR=value pairs), git.txt (commit, branch and any
# uncommitted changes) and slurm_<jobid>.out (the log). Each submission is also appended to output/runs.tsv.
#
# A tag that already has a run directory is refused, because the job scripts resume from partial
# outputs there; pass RESUME=1 to resubmit into that directory on purpose.
set -euo pipefail
cd "$(dirname "$0")/.." # repo root

if [ $# -lt 3 ]; then
    sed -n '2,13p' "$0"
    exit 1
fi
exp=$1 tag=$2 script=$3
shift 3
vars=()
sbatch_opts=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        sbatch_opts=("$@")
        break
    fi
    [[ "$1" == *=* ]] || { echo "expected VAR=value, got '$1'" >&2; exit 1; }
    vars+=("$1")
    shift
done

existing=$(ls -d output/"$exp"/*_"$tag" 2>/dev/null | tail -1 || true)
if [ -n "$existing" ] && [ "${RESUME:-0}" != 1 ]; then
    echo "run directory $existing already exists; pick another tag, or RESUME=1 to resubmit into it" >&2
    exit 1
fi
run_dir=${existing:-output/$exp/$(date +%Y-%m-%d)_$tag}
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)

cp "$script" "$run_dir/job.sh"
: > "$run_dir/params.env"
for kv in ${vars[@]+"${vars[@]}"}; do
    printf '%s=%q\n' "${kv%%=*}" "${kv#*=}" >> "$run_dir/params.env"
done
{
    echo "submitted: $(date '+%F %T')  script: $script"
    echo "branch: $(git branch --show-current)  commit: $(git rev-parse HEAD)"
    echo "uncommitted changes:"
    git status --porcelain
    git diff HEAD
} > "$run_dir/git.txt"

job_id=$(sbatch --parsable --job-name="$exp-$tag" --output="$run_dir/slurm_%j.out" \
    --export=ALL,RUN_DIR="$run_dir" ${sbatch_opts[@]+"${sbatch_opts[@]}"} "$run_dir/job.sh")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$job_id" "${run_dir#$PWD/}" "$script" "${vars[*]:-}" >> output/runs.tsv
echo "submitted job $job_id -> ${run_dir#$PWD/}"
