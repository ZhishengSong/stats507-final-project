"""BERT-based text classification model."""

from __future__ import annotations

from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification

from .base import BaseClassifier


class BertClassifier(BaseClassifier):
    """Fine-tunable BERT classifier for text-only baselines."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            labels=batch.get("labels"),
        )
        return {"loss": outputs.loss, "logits": outputs.logits}


def build_bert_model(**kwargs) -> BertClassifier:
    """Factory wrapper for :class:`BertClassifier`."""

    return BertClassifier(**kwargs)

