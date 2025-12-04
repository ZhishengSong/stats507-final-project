#!/bin/bash
#SBATCH --job-name=vilt_ablation
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

# ==========================================
# 3. Clear institution-level site-packages so we only rely on hm_env
# Ensures Python does not fall back to unexpected shared libraries.
unset PYTHONPATH
# ==========================================

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs checkpoints logs/training logs/predictions logs/jobs

TRAIN_LOG_DIR="logs/training"
PRED_DIR="logs/predictions"

export HF_DATASETS_CACHE="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
export HF_HOME="/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}"

EPOCHS=10
LR=5e-5
WARMUP=0.1
WEIGHT_DECAY=0.05
NUM_WORKERS=0

echo "========================================================================"
echo "ViLT fine-tuning strategy ablation"
echo "========================================================================"
echo "Shared config: epochs=$EPOCHS, lr=$LR, warmup=$WARMUP, wd=$WEIGHT_DECAY"
echo ""

run_vilt() {
    local label=$1
    shift
    echo "------------------------------------------------------------------------"
    echo "Running configuration: ${label}"
    echo "------------------------------------------------------------------------"
    python -m train.run_finetune \
        --model_type vilt \
        --num_train_epochs $EPOCHS \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate $LR \
        --warmup_ratio $WARMUP \
        --weight_decay $WEIGHT_DECAY \
        --max_text_length 40 \
        --output_dir "checkpoints/vilt_${label}" \
        --best_checkpoint_name best.pt \
        --log_file "${TRAIN_LOG_DIR}/vilt_${label}.log" \
        --do_test \
        --save_predictions \
        --predictions_dir "${PRED_DIR}/vilt_${label}" \
        --num_workers $NUM_WORKERS \
        --cache_dir "${HF_DATASETS_CACHE}" \
        --use_amp \
        "$@"

    mv "${PRED_DIR}/vilt_${label}/test_predictions.csv" "${PRED_DIR}/vilt_${label}_test_predictions.csv" 2>/dev/null || true
    echo ""
    echo "✓ Configuration ${label} finished"
    echo ""
}

# 1. 全量微调
run_vilt "ft_full"

# 2. 冻结视觉分支
run_vilt "freeze_vision" --freeze_vilt_vision

# 3. 冻结文本分支
run_vilt "freeze_text" --freeze_vilt_text

# 4. Train only the classification head (freeze both vision and text backbones)
run_vilt "head_only" --freeze_vilt_vision --freeze_vilt_text

echo "========================================================================"
echo "ViLT fine-tuning strategy comparison completed."
echo "Results located at:"
echo "  - ${PRED_DIR}/vilt_ft_full_test_predictions.csv"
echo "  - ${PRED_DIR}/vilt_freeze_vision_test_predictions.csv"
echo "  - ${PRED_DIR}/vilt_freeze_text_test_predictions.csv"
echo "  - ${PRED_DIR}/vilt_head_only_test_predictions.csv"
echo "========================================================================"

