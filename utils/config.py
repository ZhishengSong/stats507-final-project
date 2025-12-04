"""Configuration file parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML configuration file."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = file_path.suffix.lower()
    content = file_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs.")
        return yaml.safe_load(content)

    if suffix == ".json":
        return json.loads(content)

    raise ValueError(f"Unsupported config format: {suffix}")

