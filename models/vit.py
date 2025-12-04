"""ViT-based image classification model."""

from __future__ import annotations

from typing import Dict

import torch
from transformers import AutoModelForImageClassification

from .base import BaseClassifier


class VitClassifier(BaseClassifier):
    """Fine-tunable ViT classifier for image-only baselines."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            labels=batch.get("labels"),
        )
        return {"loss": outputs.loss, "logits": outputs.logits}


def build_vit_model(**kwargs) -> VitClassifier:
    """Factory wrapper for :class:`VitClassifier`."""

    return VitClassifier(**kwargs)

