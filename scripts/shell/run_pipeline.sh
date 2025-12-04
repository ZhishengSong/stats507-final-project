#!/bin/bash
# Sequential pipeline to fine-tune/evaluate ViLT/BERT/ViT and optional zero-shot.

set -euo pipefail

# Resolve the repo root (one level up from scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs checkpoints
mkdir -p logs/training logs/metrics logs/predictions logs/jobs

TRAIN_LOG_DIR="logs/training"
METRICS_DIR="logs/metrics"
PREDICTIONS_DIR="logs/predictions"

TRAIN_MODELS=${TRAIN_MODELS:-"vilt bert vit"}
NUM_EPOCHS=${NUM_EPOCHS:-5}
VILT_BATCH=${VILT_BATCH:-16}
BERT_BATCH=${BERT_BATCH:-32}
VIT_BATCH=${VIT_BATCH:-32}
EVAL_BATCH=${EVAL_BATCH:-32}
USE_AMP=${USE_AMP:-1}
RUN_ZERO_SHOT=${RUN_ZERO_SHOT:-0}
ZERO_SHOT_SAMPLES=${ZERO_SHOT_SAMPLES:-200}

bool_flag() {
    local flag_name=$1
    local value=$2
    if [[ "${value}" == "1" ]]; then
        echo "--${flag_name}"
    fi
}

run_finetune() {
    local model=$1
    local batch=$2
    local extra_args=$3
    local ckpt_dir="checkpoints/${model}"
    local model_pred_dir="${PREDICTIONS_DIR}/${model}"
    
    # Set cache directory for HuggingFace datasets
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache}"
    export HF_HOME="${HF_HOME:-/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_home}"
    mkdir -p "${HF_DATASETS_CACHE}" "${HF_HOME}" "${model_pred_dir}"
    
    echo "==== Training ${model} ===="
    echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
    python -m train.run_finetune \
        --model_type "${model}" \
        --num_train_epochs "${NUM_EPOCHS}" \
        --train_batch_size "${batch}" \
        --eval_batch_size "${EVAL_BATCH}" \
        --output_dir "${ckpt_dir}" \
        --best_checkpoint_name "best.pt" \
        --do_test \
        --save_predictions \
        --num_workers 0 \
        --cache_dir "${HF_DATASETS_CACHE}" \
        --log_file "${TRAIN_LOG_DIR}/${model}_finetune.log" \
        --predictions_dir "${model_pred_dir}" \
        $(bool_flag use_amp "${USE_AMP}") \
        ${extra_args}
}

run_eval_step() {
    local model=$1
    local ckpt_path="checkpoints/${model}/best.pt"
    echo "==== Evaluating ${model} from ${ckpt_path} ===="
    python -m eval.run_eval \
        --model_type "${model}" \
        --checkpoint_path "${ckpt_path}" \
        --split test \
        --save_predictions \
        --log_file "${TRAIN_LOG_DIR}/eval_${model}.log" \
        --predictions_path "${PREDICTIONS_DIR}/${model}_test_predictions.csv" \
        --metrics_path "${METRICS_DIR}/${model}_metrics.json"
}

run_zero_shot() {
    echo "==== Running Qwen-VL zero-shot on ${ZERO_SHOT_SAMPLES} samples ===="
    python -m zero_shot.run_qwenvl \
        --max_samples "${ZERO_SHOT_SAMPLES}" \
        --save_predictions \
        --log_file "${TRAIN_LOG_DIR}/zero_shot_eval.log" \
        --predictions_path "${PREDICTIONS_DIR}/zero_shot_predictions.csv" \
        --metrics_path "${METRICS_DIR}/zero_shot_metrics.json"
}

for model in ${TRAIN_MODELS}; do
    case "${model}" in
        vilt)
            run_finetune "vilt" "${VILT_BATCH}" ""
            run_eval_step "vilt"
            ;;
        bert)
            run_finetune "bert" "${BERT_BATCH}" ""
            run_eval_step "bert"
            ;;
        vit)
            run_finetune "vit" "${VIT_BATCH}" ""
            run_eval_step "vit"
            ;;
        *)
            echo "Unknown model ${model}, skipping." >&2
            ;;
    esac
done

if [[ "${RUN_ZERO_SHOT}" == "1" ]]; then
    run_zero_shot
fi

echo "==== Pipeline finished ===="
