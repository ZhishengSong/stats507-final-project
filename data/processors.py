"""Helpers that cache shared Hugging Face processors/tokenizers."""

from __future__ import annotations

from functools import lru_cache

from transformers import (
    AutoImageProcessor,
    BertTokenizerFast,
    ViltProcessor,
)
from transformers.image_processing_utils import BaseImageProcessor


@lru_cache(maxsize=None)
def get_vilt_processor(model_name: str = "dandelin/vilt-b32-mlm") -> ViltProcessor:
    """Return a cached ViLT processor instance."""

    return ViltProcessor.from_pretrained(model_name)


@lru_cache(maxsize=None)
def get_bert_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizerFast:
    """Return a cached BERT tokenizer instance."""

    return BertTokenizerFast.from_pretrained(model_name)


@lru_cache(maxsize=None)
def get_vit_image_processor(
    model_name: str = "google/vit-base-patch16-224",
) -> BaseImageProcessor:
    """Return a cached ViT image processor instance."""

    return AutoImageProcessor.from_pretrained(model_name)

