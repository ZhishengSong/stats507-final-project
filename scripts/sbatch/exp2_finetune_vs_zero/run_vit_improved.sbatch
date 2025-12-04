#!/bin/bash
#SBATCH --job-name=hm_vit_v2
#SBATCH --account=stats507f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

module purge
module load python/3.10.4
module load pytorch/2.0.1

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs checkpoints logs/training logs/metrics logs/predictions logs/jobs

TRAIN_LOG_DIR="logs/training"
PRED_DIR="logs/predictions"

export HF_DATASETS_CACHE="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
export HF_HOME="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

echo "Pre-downloading dataset metadata..."
python scripts/python/download_dataset.py

echo "=========================================="
echo "Training improved ViT model (v2)"
echo "=========================================="
python -m train.run_finetune \
    --model_type vit \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.05 \
    --output_dir checkpoints/vit_v2 \
    --best_checkpoint_name best_v2.pt \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/vit_v2" \
    --num_workers 0 \
    --cache_dir "${HF_DATASETS_CACHE}" \
    --log_file "${TRAIN_LOG_DIR}/vit_v2_train.log" \
    --use_amp

mv "${PRED_DIR}/vit_v2/test_predictions.csv" "${PRED_DIR}/vit_v2_test_predictions.csv" 2>/dev/null || true

echo "=========================================="
echo "ViT v2 training completed!"
echo "=========================================="

