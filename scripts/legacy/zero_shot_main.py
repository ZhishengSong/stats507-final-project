"""
Legacy zero-shot entrypoint that runs large multimodal models without finetuning.
"""

import argparse
import torch
from pathlib import Path

from utils import set_seed, setup_logger, get_device
from zero_shot import ZeroShotInferencer

# Note: zero-shot requires the dataset to return raw (image, text) pairs
from datasets import load_dataset


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Zero-shot inference runner")
    
    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="blip2",
        choices=["qwen-vl", "llava", "blip2"],
        help="Base model family"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name or path (defaults to the template for model_type)"
    )
    
    # Dataset settings
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (often 1 for zero-shot).")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap the number of samples for quick tests.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Dataset cache directory.")
    
    # Output
    parser.add_argument("--save_predictions", action="store_true", help="Persist predictions to disk.")
    parser.add_argument("--output_path", type=str, default="./zero_shot_predictions.csv", help="Prediction CSV path.")
    
    # Misc
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU execution.")
    
    return parser.parse_args()


def main():
    """Entrypoint."""
    args = parse_args()
    
    # Configure logging
    logger = setup_logger(log_file=args.log_file)
    logger.info("=" * 80)
    logger.info("Zero-shot inference")
    logger.info("=" * 80)
    
    # Initialize randomness
    set_seed(args.seed)
    
    # Pick device
    device = get_device(prefer_gpu=not args.no_gpu)
    
    # Resolve model name
    model_name_map = {
        "qwen-vl": "Qwen/Qwen-VL-Chat",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "blip2": "Salesforce/blip2-opt-2.7b"
    }
    model_name = args.model_name or model_name_map[args.model_type]
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Model name: {model_name}")
    logger.info(f"  Split: {args.split}")
    
    # Load dataset with both raw images and text
    logger.info(f"\nLoading {args.split} split...")
    dataset = load_dataset(
        "neuralcatcher/hateful_memes",
        split=args.split,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    # Limit sample count if requested
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"  Truncated to {len(dataset)} samples")
    
    logger.info(f"✓ Dataset ready with {len(dataset)} samples")
    
    # Build a minimal DataLoader wrapper
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Return raw tensors without additional preprocessing."""
        return {
            'image': [item['image'].convert('RGB') for item in batch],
            'text': [item['text'] for item in batch],
            'labels': torch.tensor([item['label'] for item in batch])
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Instantiate the zero-shot inferencer
    logger.info(f"\nLoading {args.model_type} model...")
    inferencer = ZeroShotInferencer(
        model_name=model_name,
        device=device,
        model_type=args.model_type
    )
    
    # Run inference + evaluation
    metrics, predictions_df = inferencer.evaluate(
        dataloader=dataloader,
        save_predictions=args.save_predictions,
        output_path=args.output_path
    )
    
    logger.info("\nZero-shot inference finished!")


if __name__ == "__main__":
    main()

