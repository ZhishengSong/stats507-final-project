"""Training and evaluation loops."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from eval.metrics import compute_classification_metrics


def _move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _gather_probs(logits: torch.Tensor) -> torch.Tensor:
    """Return the hateful-class probability (last logit)."""

    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    probs = torch.softmax(logits, dim=-1)
    return probs[..., -1]


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Train the model for a single epoch and return metrics."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    all_labels: List[int] = []
    all_probs: List[float] = []

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader, start=1):
        batch = _move_to_device(batch, device)
        with autocast(enabled=use_amp):
            outputs = model(batch)
            loss = outputs["loss"]
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        batch_size = batch["labels"].size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size * grad_accum_steps
        probs = _gather_probs(outputs["logits"]).detach().cpu()
        labels = batch["labels"].detach().cpu()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    metrics = compute_classification_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], List[int], List[float]]:
    """Evaluate the model and return metrics plus raw predictions."""

    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_to_device(batch, device)
            outputs = model(batch)
            loss = outputs["loss"]

            batch_size = batch["labels"].size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size if loss is not None else 0.0

            probs = _gather_probs(outputs["logits"]).cpu()
            labels = batch["labels"].cpu()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    metrics = compute_classification_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics, all_labels, all_probs

