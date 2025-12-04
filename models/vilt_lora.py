"""LoRA-enhanced ViLT classifier."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch.nn as nn

from .vilt import ViltClassifier

try:
    from peft import LoraConfig, get_peft_model
    from peft.utils.peft_types import TaskType
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "peft package is required for ViltLoRAClassifier. "
        "Install via `pip install peft`."
    ) from exc


class ViltLoRAClassifier(ViltClassifier):
    """Apply Low-Rank Adaptation (LoRA) to ViLT attention layers."""

    def __init__(
        self,
        model_name: str = "dandelin/vilt-b32-mlm",
        num_labels: int = 2,
        dropout: float = 0.1,
        target_modules: Sequence[str] | None = None,
        lora_rank: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        train_classifier_head: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, num_labels=num_labels, dropout=dropout)

        if target_modules is None:
            target_modules = ("query", "value")

        self._freeze_backbone_except_lora()
        self._inject_lora(
            target_modules=target_modules,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self._set_classifier_grad(train_classifier_head)

    def _freeze_backbone_except_lora(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _inject_lora(
        self,
        target_modules: Iterable[str],
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> None:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=list(target_modules),
        )
        self.model = get_peft_model(self.model, config)

    def _set_classifier_grad(self, enabled: bool) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = enabled

    def print_trainable_parameters(self) -> None:  # pragma: no cover - helper for logging
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable params: {trainable} / {total} ({trainable / total:.2%})")


def build_vilt_lora_model(**kwargs) -> ViltLoRAClassifier:
    return ViltLoRAClassifier(**kwargs)

