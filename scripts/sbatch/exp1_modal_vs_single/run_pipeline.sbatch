#!/bin/bash

#SBATCH --account=lsa2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=hm_pipeline
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

module purge
module load python/3.10.4
module load pytorch/2.0.1

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs logs/training logs/metrics logs/predictions logs/jobs

bash scripts/shell/run_pipeline.sh

