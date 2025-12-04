"""Zero-shot evaluation using Qwen-VL."""

from __future__ import annotations

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from eval.metrics import compute_classification_metrics, save_predictions_to_csv
from utils import get_logger, set_seed
from data.datasets import _ensure_rgb as load_rgb_image

DEFAULT_CACHE_DIR = os.environ.get("HF_DATASETS_CACHE")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot inference with Qwen-VL.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset_name", default="neuralcatcher/hateful_memes")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--prompt_template",
        default=(
            "You are a safety classifier. Determine whether the meme is hateful or non-hateful. "
            "Answer with exactly one token: 'hateful' or 'non-hateful'. Text content: {text}"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--log_file", default="logs/training/zero_shot.log")
    parser.add_argument(
        "--cache_dir",
        default=DEFAULT_CACHE_DIR,
        help="HF datasets cache directory (should contain hateful_memes_full/img).",
    )
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--predictions_path", default="logs/predictions/zero_shot_predictions.csv")
    parser.add_argument("--metrics_path", default="logs/metrics/zero_shot_metrics.json")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Maximum tokens generated per sample (lower values run faster).",
    )
    parser.add_argument(
        "--progress_path",
        default="logs/predictions/zero_shot_progress.csv",
        help="Path to store partial zero-shot predictions for resuming.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=25,
        help="Persist progress every N processed samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing progress file if present.",
    )
    return parser.parse_args()


def load_progress(path: Path) -> Dict[int, Dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    records: Dict[int, Dict[str, float]] = {}
    for row in df.itertuples():
        records[int(row.sample_idx)] = {"label": int(row.label), "prob": float(row.prob)}
    return records


def save_progress(path: Path, records: Dict[int, Dict[str, float]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sample_idx": idx, "label": entry["label"], "prob": entry["prob"]}
        for idx, entry in sorted(records.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger = get_logger("zero_shot", args.log_file)
    logger.info("Using device %s", device)

    dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    dataset = load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    progress_path = Path(args.progress_path)
    progress_records = load_progress(progress_path) if args.resume else {}
    processed_indices = set(progress_records.keys())
    if progress_records:
        logger.info("Loaded %d processed samples from %s", len(progress_records), progress_path)

    save_counter = 0

    for idx, sample in enumerate(dataset):
        if args.max_samples and idx >= args.max_samples:
            break
        if idx in processed_indices:
            continue
        try:
            image = load_rgb_image(sample["img"], cache_dir=args.cache_dir)
        except FileNotFoundError:
            logger.warning("Image %s missing; using gray placeholder.", sample.get("img"))
            image = Image.new("RGB", (224, 224), color="gray")
        except Exception as exc:
            logger.error("Failed to load image %s: %s", sample.get("img"), exc)
            continue
        text = sample.get("text", "")
        prompt = args.prompt_template.format(text=text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device, dtype=dtype if v.dtype == torch.float32 else None) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_text = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0]
        generation_lower = generated_text.lower()
        if "hateful" in generation_lower and "non-hateful" not in generation_lower:
            prob_hateful = 0.9
        elif "non-hateful" in generation_lower and "hateful" not in generation_lower:
            prob_hateful = 0.1
        else:
            prob_hateful = 0.5

        progress_records[idx] = {
            "label": int(sample.get("label", 0)),
            "prob": float(prob_hateful),
        }
        processed_indices.add(idx)

        save_counter += 1
        if save_counter >= args.save_every:
            save_progress(progress_path, progress_records)
            logger.info("Progress saved after %d samples.", len(processed_indices))
            save_counter = 0

    save_progress(progress_path, progress_records)
    logger.info("Final progress saved to %s", progress_path)

    ordered_indices = sorted(progress_records.keys())
    labels = [progress_records[idx]["label"] for idx in ordered_indices]
    probs = [progress_records[idx]["prob"] for idx in ordered_indices]

    metrics = compute_classification_metrics(labels, probs)
    logger.info("Zero-shot metrics: %s", json.dumps(metrics, indent=2))

    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_predictions:
        save_predictions_to_csv(labels, probs, args.predictions_path)
        logger.info("Predictions saved to %s", args.predictions_path)


if __name__ == "__main__":
    main()

