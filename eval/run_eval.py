"""Standalone evaluation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import create_dataloader
from data.collators import ImageCollator, MultimodalCollator, TextCollator
from data.processors import (
    get_bert_tokenizer,
    get_vilt_processor,
    get_vit_image_processor,
)
from eval.metrics import save_predictions_to_csv
from models import build_model
from train.loops import evaluate_model
from utils import get_logger, set_seed
from utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on Hateful Memes.")
    parser.add_argument("--model_type", choices=["vilt", "bert", "vit"], required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset_name", default="neuralcatcher/hateful_memes")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_text_length", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_file", default="logs/training/eval.log")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--predictions_path", default="logs/predictions/eval_predictions.csv")
    parser.add_argument("--metrics_path", default="logs/metrics/eval_metrics.json")
    parser.add_argument("--vilt_model_name", default="dandelin/vilt-b32-mlm")
    parser.add_argument("--bert_model_name", default="bert-base-uncased")
    parser.add_argument("--vit_model_name", default="google/vit-base-patch16-224")
    return parser.parse_args()


def get_device(no_cuda: bool) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


def build_collator(args: argparse.Namespace):
    if args.model_type == "vilt":
        processor = get_vilt_processor(args.vilt_model_name)
        return MultimodalCollator(processor=processor, max_text_length=args.max_text_length)
    if args.model_type == "bert":
        tokenizer = get_bert_tokenizer(args.bert_model_name)
        return TextCollator(tokenizer=tokenizer, max_length=args.max_text_length)
    image_processor = get_vit_image_processor(args.vit_model_name)
    return ImageCollator(image_processor=image_processor)


def build_model_from_args(args: argparse.Namespace):
    if args.model_type == "vilt":
        return build_model("vilt", model_name=args.vilt_model_name)
    if args.model_type == "bert":
        return build_model("bert", model_name=args.bert_model_name)
    return build_model("vit", model_name=args.vit_model_name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.no_cuda)
    logger = get_logger("eval", args.log_file)

    collator = build_collator(args)
    dataloader = create_dataloader(
        split=args.split,
        collate_fn=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
    )

    model = build_model_from_args(args)
    state = load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    metrics, labels, probs = evaluate_model(model, dataloader, device)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_predictions:
        save_predictions_to_csv(labels, probs, args.predictions_path)
        logger.info("Predictions saved to %s", args.predictions_path)


if __name__ == "__main__":
    main()

