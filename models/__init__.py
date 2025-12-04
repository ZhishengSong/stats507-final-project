"""Factory helpers for constructing the supported models."""

from .base import BaseClassifier  # noqa: F401
from .vilt import build_vilt_model, ViltClassifier  # noqa: F401
from .bert import build_bert_model, BertClassifier  # noqa: F401
from .vit import build_vit_model, VitClassifier  # noqa: F401


def build_model(model_type: str, **kwargs):
    """Create a model instance based on ``model_type``."""

    model_type = model_type.lower()
    if model_type == "vilt":
        return build_vilt_model(**kwargs)
    if model_type == "bert":
        return build_bert_model(**kwargs)
    if model_type == "vit":
        return build_vit_model(**kwargs)
    raise ValueError(f"Unsupported model_type={model_type}")

