"""DataLoader collators tailored for the Hateful Memes dataset."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from transformers import ViltProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class MultimodalCollator:
    """Collate batches that include both images and text."""

    def __init__(
        self,
        processor: ViltProcessor,
        max_text_length: int = 64,
    ) -> None:
        self.processor = processor
        self.max_text_length = max_text_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        sample_ids = [item["id"] for item in batch]

        encoded = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["id"] = sample_ids
        return encoded


class TextCollator:
    """Collate batches for text-only models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        sample_ids = [item["id"] for item in batch]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["id"] = sample_ids
        return encoded


class ImageCollator:
    """Collate batches for image-only models."""

    def __init__(
        self,
        image_processor: BaseImageProcessor,
    ) -> None:
        self.image_processor = image_processor

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        sample_ids = [item["id"] for item in batch]

        encoded = self.image_processor(
            images=images,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["id"] = sample_ids
        return encoded

