#!/usr/bin/env python
"""Pre-download Hateful Memes dataset metadata."""

import os
from pathlib import Path
from datasets import load_dataset

def main():
    cache_dir = os.environ.get(
        "HF_DATASETS_CACHE",
        "/scratch/stats507f25s001_class_root/stats507f25s001_class/zhisheng/hf_cache"
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset metadata to: {cache_dir}")
    
    # Load dataset metadata (not images)
    dataset = load_dataset("neuralcatcher/hateful_memes", cache_dir=cache_dir)
    
    print(f"✓ Dataset metadata downloaded!")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    print("\nNote: Images will be downloaded automatically during training.")

if __name__ == "__main__":
    main()

