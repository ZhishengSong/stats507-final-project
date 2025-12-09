# Fine-Tuning ViLT for Hateful Meme Detection

Stats 507 course project on the **Hateful Memes** dataset.

This repo implements a reproducible pipeline to study hateful meme classification with
transformer-based models. Concretely, it contains three comparative experiments:

1. **Multimodal vs. Unimodal**  
   - ViLT (image + text)  
   - BERT (text only)  
   - ViT (image only)

2. **Fine-tuned vs. Off-the-shelf & Zero-shot**  
   - Off-the-shelf backbones + random linear head (frozen encoders)  
   - Fully fine-tuned BERT / ViT / ViLT  
   - Qwen3-VL-8B-Instruct zero-shot baseline

3. **Adaptation strategies on ViLT**  
   - Full fine-tuning  
   - LoRA (parameter-efficient)  
   - BitFit (bias-only)

All experiments are run on the **school Slurm cluster** and launched via `sbatch`.

---

## 1. What’s in this repo

- **Training / models**
  - Training loop for supervised classifiers (BERT, ViT, ViLT)
  - Off-the-shelf (frozen backbone) mode
  - ViLT adapters: full FT / LoRA / BitFit

- **Zero-shot**
  - Qwen3-VL-8B-Instruct inference script for hateful / non-hateful classification

- **Slurm jobs**
  - `scripts/sbatch/` – main entry point for all experiments (submit with `sbatch`)
- **Outputs**
  - `logs/` – Slurm job logs & training logs
  - `checkpoints/` – saved model checkpoints
  - `results/` – CSV predictions & metrics (Accuracy / AUROC / Macro F1)

---

## 2. How to set up (on the school server)

```bash
# 1) clone this repo (on the login node)
git clone <this_repo_url>
cd <this_repo_name>

# 2) create / activate Python environment (conda or venv)
# example：
conda create -n hm_env python=3.10
conda activate hm_env

# 3) install dependencies
pip install -r requirements.txt
