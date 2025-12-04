"""BitFit-style ViLT classifier (bias-only tuning)."""

from __future__ import annotations

from typing import Iterable, Tuple

from .vilt import ViltClassifier


class ViltBitFitClassifier(ViltClassifier):
    """Freeze all ViLT backbone weights except bias terms."""

    def __init__(
        self,
        model_name: str = "dandelin/vilt-b32-mlm",
        num_labels: int = 2,
        dropout: float = 0.1,
        bias_keywords: Tuple[str, ...] = ("bias",),
        train_classifier_head: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, num_labels=num_labels, dropout=dropout)
        self._apply_bitfit(
            bias_keywords=bias_keywords,
            train_classifier_head=train_classifier_head,
        )

    def _apply_bitfit(
        self,
        bias_keywords: Iterable[str],
        train_classifier_head: bool,
    ) -> None:
        """Enable gradients only for bias parameters (BitFit)."""

        bias_keywords = tuple(bias_keywords)
        if not bias_keywords:
            raise ValueError("bias_keywords must contain at least one entry.")

        # Freeze entire ViLT backbone.
        for param in self.model.parameters():
            param.requires_grad = False

        # Re-enable gradients for matching bias parameters.
        for name, param in self.model.named_parameters():
            if self._matches_bias(name, bias_keywords):
                param.requires_grad = True

        # Classifier head can be optionally trained (default True).
        for param in self.classifier.parameters():
            param.requires_grad = train_classifier_head

    @staticmethod
    def _matches_bias(name: str, keywords: Tuple[str, ...]) -> bool:
        lowered = name.lower()
        return any(lowered.endswith(keyword.lower()) for keyword in keywords)


def build_vilt_bitfit_model(**kwargs) -> ViltBitFitClassifier:
    """Factory helper for BitFit ViLT."""

    return ViltBitFitClassifier(**kwargs)

