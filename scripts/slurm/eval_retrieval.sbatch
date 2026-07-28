#!/bin/bash
#SBATCH --job-name=lahn-ret
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
source ~/.bashrc
conda activate lahn-driveclip
cd "$SLURM_SUBMIT_DIR"
python evaluate_retrieval.py --config configs/talk2car_lora.yaml --checkpoint outputs/talk2car_lora/last.pt --output outputs/talk2car_retrieval.json
python evaluate_retrieval.py --config configs/bddx_cross_domain.yaml --checkpoint outputs/talk2car_lora/last.pt --output outputs/bddx_cross_domain_retrieval.json
