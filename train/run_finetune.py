"""Unified fine-tuning script for ViLT/BERT/ViT models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data import create_dataloader
from data.collators import ImageCollator, MultimodalCollator, TextCollator
from data.processors import (
    get_bert_tokenizer,
    get_vilt_processor,
    get_vit_image_processor,
)
from eval.metrics import save_predictions_to_csv
from models import build_model
from train.loops import evaluate_model, train_one_epoch
from utils import get_logger, set_seed
from utils.checkpoint import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune models on Hateful Memes.")
    parser.add_argument("--model_type", choices=["vilt", "bert", "vit"], required=True)
    parser.add_argument("--dataset_name", default="neuralcatcher/hateful_memes")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_text_length", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--best_checkpoint_name", default="best.pt")
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--log_file", default="logs/training/train.log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip fine-tuning and run zero-shot evaluation only.",
    )
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--predictions_dir", default="logs")
    parser.add_argument("--vilt_model_name", default="dandelin/vilt-b32-mlm")
    parser.add_argument("--freeze_vilt_vision", action="store_true")
    parser.add_argument("--freeze_vilt_text", action="store_true")
    parser.add_argument("--bert_model_name", default="bert-base-uncased")
    parser.add_argument("--vit_model_name", default="google/vit-base-patch16-224")
    return parser.parse_args()


def get_device(no_cuda: bool) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


def build_collator_and_loaders(args: argparse.Namespace):
    dataset_kwargs = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "cache_dir": args.cache_dir,
    }

    if args.model_type == "vilt":
        processor = get_vilt_processor(args.vilt_model_name)
        collator = MultimodalCollator(processor=processor, max_text_length=args.max_text_length)
    elif args.model_type == "bert":
        tokenizer = get_bert_tokenizer(args.bert_model_name)
        collator = TextCollator(tokenizer=tokenizer, max_length=args.max_text_length)
    else:
        image_processor = get_vit_image_processor(args.vit_model_name)
        collator = ImageCollator(image_processor=image_processor)

    train_loader = create_dataloader(
        split=args.train_split,
        collate_fn=collator,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        **dataset_kwargs,
    )

    val_loader = create_dataloader(
        split=args.val_split,
        collate_fn=collator,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        **dataset_kwargs,
    )

    test_loader = None
    if args.do_test:
        test_loader = create_dataloader(
            split=args.test_split,
            collate_fn=collator,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            **dataset_kwargs,
        )

    return collator, train_loader, val_loader, test_loader


def build_model_from_args(args: argparse.Namespace):
    if args.model_type == "vilt":
        return build_model(
            "vilt",
            model_name=args.vilt_model_name,
            freeze_vision=args.freeze_vilt_vision,
            freeze_text=args.freeze_vilt_text,
        )
    if args.model_type == "bert":
        return build_model("bert", model_name=args.bert_model_name)
    return build_model("vit", model_name=args.vit_model_name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.no_cuda)
    logger = get_logger("train", args.log_file)
    logger.info("Using device %s", device)

    _, train_loader, val_loader, test_loader = build_collator_and_loaders(args)
    model = build_model_from_args(args)
    model.to(device)

    optimizer = None
    scheduler = None
    scaler = None
    if not args.skip_training:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        num_update_steps_per_epoch = max(len(train_loader) // args.grad_accum_steps, 1)
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        warmup_steps = int(max_train_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_train_steps)
        scaler = GradScaler(enabled=args.use_amp)
    else:
        logger.info("Skip-training enabled; running zero-shot evaluation only.")

    start_epoch = 0
    best_metric = float("-inf")
    best_path = Path(args.output_dir) / args.best_checkpoint_name

    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        logger.info("Resuming from %s", args.resume_checkpoint)
        state = load_checkpoint(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state.get("epoch", 0)
        best_metric = state.get("best_metric", best_metric)

    if not args.skip_training:
        for epoch in range(start_epoch, args.num_train_epochs):
            logger.info("Epoch %d/%d", epoch + 1, args.num_train_epochs)
            train_stats = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                scheduler=scheduler,
                grad_accum_steps=args.grad_accum_steps,
                max_grad_norm=args.max_grad_norm,
                use_amp=args.use_amp,
            )
            logger.info("Train metrics: %s", json.dumps(train_stats, indent=2))

            val_metrics, _, _ = evaluate_model(model, val_loader, device)
            logger.info("Val metrics: %s", json.dumps(val_metrics, indent=2))

            current_metric = val_metrics.get("auroc", float("-inf"))
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "best_metric": best_metric,
                        "args": vars(args),
                    },
                    checkpoint_dir=args.output_dir,
                    filename=args.best_checkpoint_name,
                )
                logger.info("Saved new best checkpoint with AUROC=%.4f", best_metric)
    else:
        logger.info("Training loop skipped; evaluating pretrained weights only.")
        if val_loader is not None:
            val_metrics, _, _ = evaluate_model(model, val_loader, device)
            logger.info("Val metrics (zero-shot): %s", json.dumps(val_metrics, indent=2))

    if args.do_test and test_loader is not None:
        if not args.skip_training:
            if best_path.exists():
                logger.info("Evaluating best checkpoint on test split.")
                state = load_checkpoint(str(best_path), map_location=device)
                model.load_state_dict(state["model_state_dict"])
            else:
                logger.warning(
                    "Best checkpoint %s not found; evaluating current model weights.",
                    best_path,
                )
        else:
            logger.info("Evaluating pretrained weights on test split (zero-shot).")

        test_metrics, labels, probs = evaluate_model(model, test_loader, device)
        logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

        if args.save_predictions:
            Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
            csv_path = Path(args.predictions_dir) / "test_predictions.csv"
            save_predictions_to_csv(labels, probs, str(csv_path))
            logger.info("Saved test predictions to %s", csv_path)


if __name__ == "__main__":
    main()

