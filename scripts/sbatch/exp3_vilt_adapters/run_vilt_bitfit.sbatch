#!/bin/bash
#SBATCH --job-name=vilt_bitfit
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
# 3. Clear the campus-wide Python paths so we only use the virtualenv
# This avoids mixing pre-installed packages with our pinned versions.
unset PYTHONPATH
# ==========================================

source "$HOME/hm_env/bin/activate"

cd /home/zhisheng/stats507
mkdir -p logs checkpoints logs/training logs/metrics logs/predictions logs/jobs

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
echo "ViLT BitFit (bias-only) fine-tuning run"
echo "========================================================================"
echo "Shared config: epochs=$EPOCHS, lr=$LR, warmup=$WARMUP, wd=$WEIGHT_DECAY"
echo ""

python -m train.run_vilt_bitfit \
    --num_train_epochs $EPOCHS \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate $LR \
    --warmup_ratio $WARMUP \
    --weight_decay $WEIGHT_DECAY \
    --max_text_length 40 \
    --output_dir checkpoints/vilt_bitfit \
    --best_checkpoint_name best.pt \
    --log_file "${TRAIN_LOG_DIR}/vilt_bitfit.log" \
    --do_test \
    --save_predictions \
    --predictions_dir "${PRED_DIR}/vilt_bitfit" \
    --num_workers $NUM_WORKERS \
    --cache_dir "${HF_DATASETS_CACHE}" \
    --use_amp \
    --bias_keywords bias \
    --freeze_classifier_head

mv "${PRED_DIR}/vilt_bitfit/test_predictions.csv" "${PRED_DIR}/vilt_bitfit_test_predictions.csv" 2>/dev/null || true

echo ""
echo "✓ ViLT BitFit run complete"
echo "Results written to ${PRED_DIR}/vilt_bitfit_test_predictions.csv"
echo "========================================================================"

