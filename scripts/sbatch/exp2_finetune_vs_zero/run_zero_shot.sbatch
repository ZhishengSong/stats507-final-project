#!/bin/bash
#SBATCH --job-name=hm_zero_shot
#SBATCH --account=stats507f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

# 1. Reset modules to start from a clean slate
module purge

# 2. Load the base Python toolchain
module load python/3.10.4
module load pytorch/2.0.1

# ==========================================
# 3. Clear institution-wide PYTHONPATH so only our virtualenv is visible
# Prevents Python from importing pre-installed site-packages.
unset PYTHONPATH
# ==========================================

# 4. Activate the project virtual environment
source "$HOME/hm_env/bin/activate"

# 5. Configure Hugging Face caches so Git LFS images can be resolved
export HF_DATASETS_CACHE="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
export HF_HOME="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

# 6. Optional: print environment details to confirm the PyTorch build
echo "Debugging Environment:"
which python
python -c "import torch; print(f'Loaded Torch Version: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
echo "------------------------------------------------"

cd /home/zhisheng/stats507
mkdir -p logs logs/training logs/metrics logs/predictions logs/jobs

python -m zero_shot.run_qwenvl \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --split test \
    --cache_dir "${HF_DATASETS_CACHE}" \
    --max_samples 1500 \
    --max_new_tokens 48 \
    --progress_path logs/predictions/zero_shot_progress.csv \
    --save_every 25 \
    --resume \
    --save_predictions \
    --log_file logs/training/zero_shot_job.log \
    --predictions_path logs/predictions/zero_shot_predictions.csv \
    --metrics_path logs/metrics/zero_shot_metrics.json