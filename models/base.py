"""Base abstractions for classifiers."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class BaseClassifier(nn.Module):
    """A minimal interface shared by all classifiers."""

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subclasses must implement the forward pass."""

        raise NotImplementedError("Subclasses must implement forward().")

    def inference_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run inference with ``torch.no_grad`` convenience handling."""

        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
        return outputs

