# Running jobs

Every job is submitted through `slurm/submit.sh`, which gives it its own run directory:

```
output/<exp>/<date>_<tag>/
    job.sh            copy of the job script; this copy is what Slurm runs
    params.env        the VAR=value pairs the job read
    git.txt           branch, commit and any uncommitted changes at submission
    slurm_<jobid>.out the log
    ...               the job's results (retrieval.json, eval.txt, ...)
output/runs.tsv       one line per submission: time, job id, run dir, script, variables
```

So every result can be traced to the exact script, parameters and code that produced it, and editing a
job script never changes a run that was already submitted.

```bash
slurm/submit.sh <exp> <tag> <job script> [VAR=value ...] [-- extra sbatch options]
```

- A tag that already has a run directory is refused, because the scripts resume from partial outputs in
it (that is how a stale control run once got silently reused). Pick a new tag, or pass `RESUME=1` to
resubmit into the existing directory on purpose (e.g. after a timeout).
- Anything after `--` goes to `sbatch` and overrides the script's `#SBATCH` lines:
`-- --time=6:00:00`, `-- --gres=gpu:l40s:1 --mem=32G`, `-- --dependency=afterok:<jobid>`.
- `slurm/common.sh` is the shared setup every job sources (venv, `HF_HOME`, `DATA_ROOT`, reading `params.env`).

## oolong

| Script | What it does | Variables |
|---|---|---|
| `oolong/prepare.sh` | parquet -> retrieval JSON | `SPLIT`, `MAX_CONTEXT_LEN`, `OUT_FILE`, `OVERWRITE=1` |
| `oolong/retrieval.sh` | attention retrieval + its eval | `RETRIEVER`, `MODEL_CONFIG` (required), `DATA_FILE`, `SUBSAMPLE`, `MAX_INSTANCES`, `MAX_NEW_TOKENS`, `Q_WEIGHT`, `COT_INSTRUCTION`, `COT_FILE`, `NULL_COT=1` |
| `oolong/bm25.sh` | BM25 baselines + their eval | `COT_FILE`, `MODES`, `DATA_FILE` |
| `oolong/eval.sh` | several results side by side | `RUNS` (required), `DATA_FILE` |

The Llama-3.1-8B oolong experiments, as they would be submitted now:

```bash
LLAMA=src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
slurm/submit.sh oolong llama_qr      slurm/oolong/retrieval.sh RETRIEVER=qr_head     MODEL_CONFIG=$LLAMA
slurm/submit.sh oolong llama_control slurm/oolong/retrieval.sh RETRIEVER=cot_qr_head MODEL_CONFIG=$LLAMA MAX_NEW_TOKENS=0
slurm/submit.sh oolong llama_cot     slurm/oolong/retrieval.sh RETRIEVER=cot_qr_head MODEL_CONFIG=$LLAMA
# BM25 with the CoTs of llama_cot, once that run has finished (or: -- --dependency=afterok:<its job id>)
slurm/submit.sh oolong bm25_llama_cot slurm/oolong/bm25.sh COT_FILE=output/oolong/<date>_llama_cot/retrieval_cot.json
```

Results from before this layout stay where they were: `output/oolong_cot/`, `output/oolong_bm25/`,
`output/oolong_results/`, `output/lme_*`, with their logs in `out/`. The `slurm/lme/` and `slurm/cot/`
scripts are kept as the record of those LongMemEval runs and still use the old layout.

## Caveat: the code is installed in editable mode

`.venv` imports `qrretriever` straight from `src/`, and a job reads `exp_scripts/` when it starts. Switching
git branches or editing code while a job is queued or running changes the code that job uses. `git.txt`
records the commit at submission, not at start, so keep the working tree on the intended commit until the
job has started (and loaded the model).
