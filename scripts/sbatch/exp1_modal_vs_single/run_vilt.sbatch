#!/bin/bash
#SBATCH --job-name=hm_vilt
#SBATCH --account=stats507f25s001_class   # Course-required billing account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00                   # Allocate 4 hours for roughly 5 epochs
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

module purge
module load python/3.10.4
module load pytorch/2.0.1

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs checkpoints logs/training logs/metrics logs/predictions logs/jobs

# Set cache directories
export HF_DATASETS_CACHE="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
export HF_HOME="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

# Pre-download dataset metadata (images will be downloaded on-the-fly during training)
echo "Pre-downloading dataset metadata..."
python scripts/python/download_dataset.py

# Train ViLT only
export TRAIN_MODELS="vilt"
export RUN_ZERO_SHOT=0

bash scripts/shell/run_pipeline.sh
