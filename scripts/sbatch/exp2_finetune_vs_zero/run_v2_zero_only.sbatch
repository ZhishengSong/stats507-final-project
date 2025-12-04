#!/bin/bash
#SBATCH --job-name=hm_v2_zero_only
#SBATCH --account=stats507f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --error=logs/jobs/%x_%j.err

set -euo pipefail

module purge
module load python/3.10.4
module load pytorch/2.0.1

unset PYTHONPATH

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs checkpoints logs/training logs/metrics logs/predictions logs/jobs

TRAIN_LOG_DIR="logs/training"
PRED_DIR="logs/predictions"

export HF_DATASETS_CACHE="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
export HF_HOME="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

echo "=============================================="
echo "Zero-shot evaluation for ViLT / BERT / ViT only"
echo "=============================================="

echo ""
echo "[1/3] ViLT zero-shot"
python -m train.run_finetune \
    --model_type vilt \
    --skip_training \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --max_text_length 40 \
    --output_dir checkpoints/vilt_v2_zero_only \
    --best_checkpoint_name zero_shot.pt \
    --log_file "${TRAIN_LOG_DIR}/vilt_v2_zero_only.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/vilt_v2_zero_only" \
    --num_workers 0 \
    --cache_dir "${HF_DATASETS_CACHE}"
mv "${PRED_DIR}/vilt_v2_zero_only/test_predictions.csv" \
   "${PRED_DIR}/vilt_v2_zero_only_test_predictions.csv" 2>/dev/null || true

echo ""
echo "[2/3] BERT zero-shot"
python -m train.run_finetune \
    --model_type bert \
    --skip_training \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_text_length 64 \
    --output_dir checkpoints/bert_v2_zero_only \
    --best_checkpoint_name zero_shot.pt \
    --log_file "${TRAIN_LOG_DIR}/bert_v2_zero_only.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/bert_v2_zero_only" \
    --num_workers 0 \
    --cache_dir "${HF_DATASETS_CACHE}"
mv "${PRED_DIR}/bert_v2_zero_only/test_predictions.csv" \
   "${PRED_DIR}/bert_v2_zero_only_test_predictions.csv" 2>/dev/null || true

echo ""
echo "[3/3] ViT zero-shot"
python -m train.run_finetune \
    --model_type vit \
    --skip_training \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --output_dir checkpoints/vit_v2_zero_only \
    --best_checkpoint_name zero_shot.pt \
    --log_file "${TRAIN_LOG_DIR}/vit_v2_zero_only.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/vit_v2_zero_only" \
    --num_workers 0 \
    --cache_dir "${HF_DATASETS_CACHE}"
mv "${PRED_DIR}/vit_v2_zero_only/test_predictions.csv" \
   "${PRED_DIR}/vit_v2_zero_only_test_predictions.csv" 2>/dev/null || true

echo ""
echo "All zero-shot evaluations complete."

