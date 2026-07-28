#!/bin/bash
#SBATCH --job-name=lahn-loc
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
source ~/.bashrc
conda activate lahn-driveclip
cd "$SLURM_SUBMIT_DIR"
CONFIG="${1:-configs/kitti_localization.yaml}"
NAME="${2:-kitti}"
python evaluate_localization.py --config "$CONFIG" --checkpoint outputs/talk2car_lora/last.pt --method gscorecam --gscorecam-path gScoreCAM --output "outputs/${NAME}_localization.json"
