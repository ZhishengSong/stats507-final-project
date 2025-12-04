"""Helpers for saving and loading checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    """Persist a checkpoint dictionary and return its path."""

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    torch.save(state, file_path)
    return str(file_path)


def load_checkpoint(
    checkpoint_path: str,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    return torch.load(checkpoint_path, map_location=map_location)

