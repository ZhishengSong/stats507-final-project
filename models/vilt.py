"""ViLT-based multimodal classifier."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from transformers import ViltConfig, ViltModel

from .base import BaseClassifier


class ViltClassifier(BaseClassifier):
    """Binary classifier that fine-tunes a pre-trained ViLT backbone."""

    def __init__(
        self,
        model_name: str = "dandelin/vilt-b32-finetuned",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_vision: bool = False,
        freeze_text: bool = False,
    ) -> None:
        super().__init__()
        self.config = ViltConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.model = ViltModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        if freeze_vision:
            self._set_requires_grad(self.model.vision_model, False)
        if freeze_text:
            self._set_requires_grad(self.model.text_transformer, False)

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )
        pooled = outputs.pooler_output  # [batch, hidden]
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if "labels" in batch:
            loss = self.loss_fn(logits, batch["labels"])

        return {"loss": loss, "logits": logits}


def build_vilt_model(**kwargs) -> ViltClassifier:
    """Factory wrapper for :class:`ViltClassifier`."""

    return ViltClassifier(**kwargs)

