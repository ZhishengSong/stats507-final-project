#!/bin/bash
#SBATCH --job-name=hm_all_v2
#SBATCH --account=stats507f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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
METRICS_DIR="logs/metrics"
PRED_DIR="logs/predictions"

export HF_DATASETS_CACHE="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
export HF_HOME="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

# Shared training hyper-parameters (for apples-to-apples comparison)
EPOCHS=10
LR=5e-5
WARMUP=0.1
WEIGHT_DECAY=0.05
NUM_WORKERS=0

echo "========================================================================"
echo "Fair comparison: all three models share the same v2 configuration"
echo "========================================================================"
echo "Shared config: epochs=$EPOCHS, lr=$LR, warmup=$WARMUP, wd=$WEIGHT_DECAY"
echo ""

# 1. Train ViLT
echo "========================================================================"
echo "[1/3] Training ViLT v2"
echo "========================================================================"
python -m train.run_finetune \
    --model_type vilt \
    --num_train_epochs $EPOCHS \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate $LR \
    --warmup_ratio $WARMUP \
    --weight_decay $WEIGHT_DECAY \
    --max_text_length 40 \
    --output_dir checkpoints/vilt_v2 \
    --best_checkpoint_name best.pt \
    --log_file "${TRAIN_LOG_DIR}/vilt_v2_train.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/vilt_v2" \
    --num_workers $NUM_WORKERS \
    --cache_dir "${HF_DATASETS_CACHE}" \
    --use_amp

# Rename the output for easier downstream consumption
mv "${PRED_DIR}/vilt_v2/test_predictions.csv" "${PRED_DIR}/vilt_v2_test_predictions.csv" 2>/dev/null || true

echo ""
echo "✓ ViLT v2 training complete"
echo ""

# 2. Train BERT
echo "========================================================================"
echo "[2/3] Training BERT v2"
echo "========================================================================"
python -m train.run_finetune \
    --model_type bert \
    --num_train_epochs $EPOCHS \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate $LR \
    --warmup_ratio $WARMUP \
    --weight_decay $WEIGHT_DECAY \
    --max_text_length 64 \
    --output_dir checkpoints/bert_v2 \
    --best_checkpoint_name best.pt \
    --log_file "${TRAIN_LOG_DIR}/bert_v2_train.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/bert_v2" \
    --num_workers $NUM_WORKERS \
    --cache_dir "${HF_DATASETS_CACHE}" \
    --use_amp

# Rename the output for consistency
mv "${PRED_DIR}/bert_v2/test_predictions.csv" "${PRED_DIR}/bert_v2_test_predictions.csv" 2>/dev/null || true

echo ""
echo "✓ BERT v2 training complete"
echo ""

# 3. Train ViT
echo "========================================================================"
echo "[3/3] Training ViT v2"
echo "========================================================================"
python -m train.run_finetune \
    --model_type vit \
    --num_train_epochs $EPOCHS \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate $LR \
    --warmup_ratio $WARMUP \
    --weight_decay $WEIGHT_DECAY \
    --output_dir checkpoints/vit_v2 \
    --best_checkpoint_name best.pt \
    --log_file "${TRAIN_LOG_DIR}/vit_v2_train.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/vit_v2" \
    --num_workers $NUM_WORKERS \
    --cache_dir "${HF_DATASETS_CACHE}" \
    --use_amp

# 重命名输出文件
mv "${PRED_DIR}/vit_v2/test_predictions.csv" "${PRED_DIR}/vit_v2_test_predictions.csv" 2>/dev/null || true

echo ""
echo "✓ ViT v2 完成"
echo ""

echo "========================================================================"
echo "所有v2训练完成！"
echo "========================================================================"
echo "结果文件："
echo "  - ${PRED_DIR}/vilt_v2_test_predictions.csv"
echo "  - ${PRED_DIR}/bert_v2_test_predictions.csv"
echo "  - ${PRED_DIR}/vit_v2_test_predictions.csv"
echo "========================================================================"

