# Hateful Memes Multimodal Benchmark

This repository hosts our end-to-end benchmark for the Hateful Memes challenge.  
It contains reproducible training recipes for ViLT (image + text), BERT (text only), ViT (image only), plus a Qwen‑VL zero-shot baseline. All experiments share splits, optimizer settings, and logging so results remain comparable.

---

## Repository structure (high level)

- `data/`, `models/`, `train/` – dataset adapters, model definitions, and shared train/eval loops.  
- `scripts/` – automation:
  - `scripts/shell/` bash helpers (environment setup, pipeline runner)  
  - `scripts/python/` utilities (dataset download, analysis)  
  - `scripts/sbatch/exp*/` grouped Slurm jobs for each experiment family  
- `zero_shot/` – Qwen‑VL inference entrypoint  
- `logs/` + `results/` – runtime logs and final metrics/predictions/plots

---

## Quick start

1. **Create / activate the experiment environment**
   ```bash
   bash scripts/shell/setup_hm_env.sh
   ```
   (If the cluster script cannot be used, manually `module purge`, load `python/3.10.4` + `pytorch/2.0.1`, `unset PYTHONPATH`, create/activate `~/hm_env`, and `pip install -r requirements.txt`.)

2. **Cache the dataset and checkpoints**
   ```bash
   export HF_DATASETS_CACHE=/scratch/your_path/hf_cache
   export HF_HOME=/scratch/your_path/hf_home
   python scripts/python/download_dataset.py
   ```

3. **Train / evaluate**
   ```bash
   # run all three v2 baselines with consistent hyper-parameters
   sbatch scripts/sbatch/run_all_v2.sbatch
   # or submit a single model run
   sbatch scripts/sbatch/exp2_finetune_vs_zero/run_vilt_improved.sbatch
   ```
   Each job writes training logs under `logs/training/`, best checkpoints under `checkpoints/<model>`, and evaluation CSV / metrics under `results/`.

4. **Manual evaluation of a checkpoint**
   ```bash
   python -m eval.run_eval \
     --model_type vilt \
     --checkpoint_path checkpoints/vilt_v2/best.pt \
     --split test \
     --save_predictions
   ```

5. **Zero-shot Qwen-VL baseline**
   ```bash
   python -m zero_shot.run_qwenvl \
     --model_name Qwen/Qwen3-VL-8B-Instruct \
     --split test \
     --save_predictions \
     --resume \
     --max_new_tokens 48
   ```

---

## Key results (test split)

| Model | Modality        | Accuracy | AUROC  | Macro F1 |
|-------|-----------------|----------|--------|----------|
| ViLT  | Image + Text    | **0.687** | **0.7395** | **0.6434** |
| BERT  | Text only       | 0.620    | 0.6509 | 0.5182 |
| ViT   | Image only      | 0.586    | 0.5623 | 0.5314 |
| Qwen-VL (zero-shot) | Image + Text | 0.660 | 0.6460 | 0.3976 |

**Takeaways**
- Multi-modal fusion (ViLT) delivers +13.6% AUROC vs. the text-only baseline and +31.5% vs. the image-only baseline.
- Text carries more signal than images for hateful memes, but ignoring either modality leaves accuracy on the table.
- The zero-shot Qwen-VL run establishes a strong reference point for future larger-model experiments.

All metrics, CSV predictions, and plots live inside `results/`. Logs for every submitted job stay under `logs/jobs/<model>/…` for debugging.

---

