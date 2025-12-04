"""Quick sanity check to verify that the Hateful Memes dataset can load images."""

from data.datasets import HatefulMemesDataset
from PIL import Image


def test_dataset() -> bool:
    print("Testing dataset loading...")

    dataset = HatefulMemesDataset(
        split="train",
        dataset_name="neuralcatcher/hateful_memes",
        cache_dir=None,
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    try:
        sample = dataset[0]
        print("✓ Successfully loaded sample 0")
        print(f"  - Image type: {type(sample['image'])}")
        if isinstance(sample["image"], Image.Image):
            print(f"  - Image size: {sample['image'].size}")
        text = sample["text"]
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"  - Text: {preview}")
        print(f"  - Label: {sample['label']}")
        return True
    except Exception as exc:
        print(f"✗ Failed to load sample 0: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset()
    raise SystemExit(0 if success else 1)


