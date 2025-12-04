"""Utility helpers for loading the Hateful Memes dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from datasets import Features, Image as HFImage
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SplitLike = str

SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


def _resolve_split(split: SplitLike) -> str:
    key = split.lower()
    if key not in SPLIT_ALIASES:
        return split
    return SPLIT_ALIASES[key]


def _ensure_rgb(image: Any, cache_dir: Optional[str] = None) -> Image.Image:
    """Convert various image formats to RGB PIL Image.
    
    Args:
        image: Can be PIL Image, file path string, dict with 'path'/'bytes', URL, or bytes.
        cache_dir: Optional cache directory to look for images
    
    Returns:
        RGB PIL Image.
    """
    import requests
    from io import BytesIO
    
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    
    if isinstance(image, str):
        # Try as local file path first
        if Path(image).exists():
            return Image.open(image).convert("RGB")
        
        # If a cache_dir is provided, prefer local copies before hitting the network
        if cache_dir:
            # Look for the sample inside the git-cloned hateful_memes_full tree
            cached_path = Path(cache_dir) / "hateful_memes_full" / image
            if cached_path.exists():
                return Image.open(cached_path).convert("RGB")
            
            # Fallback: check the directory used for URL-downloaded images
            cached_path2 = Path(cache_dir) / "hateful_memes_images" / image
            if cached_path2.exists():
                return Image.open(cached_path2).convert("RGB")
            
            # Try direct path in cache_dir
            cached_path3 = Path(cache_dir) / image
            if cached_path3.exists():
                return Image.open(cached_path3).convert("RGB")
        
        # Try as URL (http/https)
        if image.startswith(('http://', 'https://')):
            try:
                response = requests.get(image, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to download image from {image}: {e}")
        
        # If it looks like a HF dataset path, try multiple sources
        # For hateful_memes, images might be available from the dataset repo or Facebook
        if image.startswith('img/'):
            # Extract just the filename
            img_filename = Path(image).name
            
            # Try multiple possible URLs
            urls_to_try = [
                f"https://huggingface.co/datasets/neuralcatcher/hateful_memes/resolve/main/{image}",
                f"https://huggingface.co/datasets/limjiayi/hateful_memes_expanded/resolve/main/{image}",
                # Facebook's original URL pattern (if accessible)
                f"https://github.com/facebookresearch/mmf/raw/main/projects/hateful_memes/data/{image}",
            ]
            
            last_error = None
            for url in urls_to_try:
                try:
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    last_error = e
                    continue
            
            raise FileNotFoundError(f"Image not found at any URL. Last error: {last_error}. Tried: {urls_to_try}")
        
        raise FileNotFoundError(f"Image file not found: {image}")
    
    if isinstance(image, dict):
        # Handle dict with 'path' or 'bytes' key
        if "path" in image and image["path"]:
            return _ensure_rgb(image["path"], cache_dir)
        if "bytes" in image and image["bytes"]:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
    
    if isinstance(image, bytes):
        # Handle raw bytes
        return Image.open(BytesIO(image)).convert("RGB")
    
    raise TypeError(f"Unsupported image type: {type(image)}. Value: {image}")


class HatefulMemesDataset(Dataset):
    """PyTorch dataset wrapper for Hateful Memes.

    Args:
        split: Dataset split (train/validation/test).
        dataset_name: Hugging Face dataset identifier.
        dataset_config: Optional subset name.
        cache_dir: Optional cache directory for HF datasets.
        image_transform: Optional single-sample image transform.
        text_transform: Optional single-sample text transform.
        fields_to_keep: Additional raw fields to preserve in each sample.
    """

    def __init__(
        self,
        split: SplitLike,
        dataset_name: str = "neuralcatcher/hateful_memes",
        dataset_config: Optional[str] = None,
        cache_dir: Optional[str] = None,
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
        text_transform: Optional[Callable[[str], str]] = None,
        fields_to_keep: Optional[Sequence[str]] = None,
        streaming: bool = False,
    ) -> None:
        self.split = _resolve_split(split)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.cache_dir = cache_dir
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.fields_to_keep = list(fields_to_keep) if fields_to_keep else []
        self.streaming = streaming

        self._dataset = self._load_dataset()

    def _load_dataset(self) -> Iterable[Dict[str, Any]]:
        if self.streaming:
            return load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=True,
            )
        
        # Load dataset
        hf_dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        
        # Critical: Set format to load images directly as PIL objects in worker processes
        # This prevents FileNotFoundError when paths are relative
        if "img" in hf_dataset.column_names:
            # Ensure images are decoded immediately when accessed
            hf_dataset = hf_dataset.with_format("python")
        
        return hf_dataset

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Streaming dataset has no __len__.")
        assert isinstance(self._dataset, HFDataset)
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.streaming:
            raise TypeError("Streaming dataset does not support random access.")

        assert isinstance(self._dataset, HFDataset)
        
        try:
            sample = self._dataset[index]
        except Exception as e:
            raise RuntimeError(f"Failed to load sample at index {index}: {e}") from e
        
        # Extract and convert image with better error handling
        try:
            img_data = sample.get("img") or sample.get("image")
            if img_data is None:
                raise ValueError(f"No image field found in sample {index}. Available keys: {list(sample.keys())}")
            image = _ensure_rgb(img_data, cache_dir=self.cache_dir)
        except FileNotFoundError:
            # Skip samples with missing images (use a placeholder gray image)
            image = Image.new("RGB", (224, 224), color="gray")
            print(f"Warning: Image not found for sample {index}, using placeholder.")
        except Exception as e:
            raise RuntimeError(f"Failed to load/convert image at index {index}. Image data type: {type(sample.get('img'))}. Error: {e}") from e
        
        text = str(sample.get("text", ""))
        label = int(sample.get("label", 0))
        sample_id = sample.get("id", index)

        if self.image_transform:
            image = self.image_transform(image)
        if self.text_transform:
            text = self.text_transform(text)

        result: Dict[str, Any] = {
            "image": image,
            "text": text,
            "label": label,
            "id": sample_id,
        }

        for field in self.fields_to_keep:
            if field in sample:
                result[field] = sample[field]

        return result


def create_dataloader(
    split: SplitLike,
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Build a DataLoader with common defaults."""

    dataset = HatefulMemesDataset(split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split.lower().startswith("train")

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

