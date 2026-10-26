# Module stack and venv for vLLM jobs. Source it instead of slurm/common.sh's .venv:
# vLLM 0.32 needs torch 2.13 and transformers 5.x, while .venv stays on transformers 4.51.3 for the custom models.
# Stack from https://docs.alliancecan.ca/wiki/VLLM (the wheelhouse vllm >= 0.22 is built with cuda/13.2), except
# python/3.14: vllm 0.32.0 -> flashinfer-python 0.7.0.post1 -> nccl-extensions, which the wheelhouse only has for cp314.
# slurm/setup_vllm_env.sh builds the venv with this same stack; a venv breaks under a different python module.

module load StdEnv/2023 gcc/12.3 cuda/13.2 opencv/4.13.0 python/3.14.2
VLLM_ENV=${VLLM_ENV:-$SCRATCH/venvs/vllm}
