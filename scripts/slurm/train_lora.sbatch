#!/bin/bash
#SBATCH --job-name=lahn-lora
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
source ~/.bashrc
conda activate lahn-driveclip
cd "$SLURM_SUBMIT_DIR"
python train.py --config configs/talk2car_lora.yaml
