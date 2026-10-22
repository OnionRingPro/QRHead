#!/bin/bash
#SBATCH --job-name=prepare-oolong
#SBATCH --account=aip-xiye17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/project/aip-xiye17/chenqi3/QRHead
#SBATCH --output=./out/%x_%j.out

module load StdEnv/2023 python/3.11.5 arrow/22.0.0 
source .venv/bin/activate

python preprocess/data_preparation/prepare_oolong.py \
    --raw_dir $SCRATCH/qrhead_data/raw/oolong-synth \
    --output_file $SCRATCH/qrhead_data/data/oolong_data/synth_validation.json \
    --split validation
