#!/bin/bash
#SBATCH --job-name=hm_vit
#SBATCH --account=stats507f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

module purge
module load python/3.10.4
module load pytorch/2.0.1

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs checkpoints logs/training logs/metrics logs/predictions logs/jobs

# 只训练+评估 ViT
export TRAIN_MODELS="vit"
export RUN_ZERO_SHOT=0

bash scripts/shell/run_pipeline.sh
