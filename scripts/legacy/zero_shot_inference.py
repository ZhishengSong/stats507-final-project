"""
Zero-shot inference helpers for large vision-language models.

These utilities evaluate pre-trained LMMs (e.g., Qwen-VL, LLaVA) on Hateful
Memes as a baseline against finetuned classifiers.

Example:
    python -m zero_shot.inference --model qwen-vl --output outputs/zero_shot_predictions.csv
"""

import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from io import BytesIO
import csv
import re

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from eval.evaluator import compute_metrics, print_classification_report


class ZeroShotInference:
    """
    Base class for large-model zero-shot inference.

    Provides a unified interface across multiple vision-language models.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize the inference wrapper.

        Args:
            device: torch.device used for execution.
        """
        self.device = device
        self.model = None
        self.processor = None
    
    def predict(self, image: Image.Image, text: str) -> Tuple[int, float]:
        """
        Predict on a single sample.

        Args:
            image: PIL Image.
            text: Meme caption text.

        Returns:
            Tuple of (predicted label, hateful probability).
        """
        raise NotImplementedError("Subclasses must implement predict().")
    
    def predict_batch(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> Tuple[List[int], List[float]]:
        """
        Predict on a list of samples.

        The default implementation iterates one-by-one; subclasses can
        override to support true batching.

        Args:
            images: List of PIL images.
            texts: List of meme texts.

        Returns:
            Tuple of (labels, probabilities).
        """
        predictions = []
        probabilities = []
        
        for image, text in zip(images, texts):
            pred, prob = self.predict(image, text)
            predictions.append(pred)
            probabilities.append(prob)
        
        return predictions, probabilities


class QwenVLInference(ZeroShotInference):
    """
    Zero-shot inferencer backed by Qwen-VL-Chat.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",
        device: torch.device = None,
    ):
        """
        Initialize the Qwen-VL inferencer.

        Args:
            model_name: Hugging Face model identifier.
            device: torch.device override.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device)
        
        print(f"Loading Qwen-VL model: {model_name}")
        print("⚠ The checkpoint is large; loading may take a few minutes...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).eval()
            
            print("✓ Qwen-VL model ready")
            
        except Exception as e:
            print(f"✗ Failed to load Qwen-VL: {e}")
            print("Hint: ensure transformers>=4.35.0 is installed and VRAM is sufficient.")
            raise
    
    def predict(self, image: Image.Image, text: str) -> Tuple[int, float]:
        """
        Run zero-shot prediction with Qwen-VL.
        """
        # Qwen-VL expects an on-disk image path; write a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f.name)
            image_path = f.name
        
        # Build the prompt
        prompt = f"""Look at this meme image and its text caption: "{text}"

Is this meme hateful (containing hate speech, discrimination, or offensive content targeting specific groups)?

Please answer with just "Yes" or "No", followed by a confidence score from 0 to 100.
Format: [Yes/No], [confidence]"""

        try:
            query = self.tokenizer.from_list_format([
                {"image": image_path},
                {"text": prompt},
            ])
            
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
            
            # Parse the response
            response_lower = response.lower().strip()
            
            # Extract Yes/No and a confidence estimate
            is_hateful = "yes" in response_lower[:20]  # focus on the beginning
            
            # Extract numeric confidence if present
            confidence = 0.5
            numbers = re.findall(r'\d+', response)
            if numbers:
                conf_value = int(numbers[-1])
                if 0 <= conf_value <= 100:
                    confidence = conf_value / 100.0
            
            prediction = 1 if is_hateful else 0
            probability = confidence if is_hateful else (1 - confidence)
            
            return prediction, probability
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.5
        finally:
            # Clean up temporary asset
            import os
            if os.path.exists(image_path):
                os.remove(image_path)


class LLaVAInference(ZeroShotInference):
    """
    Zero-shot inferencer powered by LLaVA.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: torch.device = None,
    ):
        """
        Initialize the LLaVA wrapper.

        Args:
            model_name: Hugging Face model identifier.
            device: torch.device override.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device)
        
        print(f"Loading LLaVA model: {model_name}")
        print("⚠ This checkpoint is large; loading may take a few minutes...")
        
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).eval()
            
            print("✓ LLaVA model ready")
            
        except Exception as e:
            print(f"✗ Failed to load LLaVA: {e}")
            print("Hint: install the latest transformers release and ensure VRAM headroom.")
            raise
    
    def predict(self, image: Image.Image, text: str) -> Tuple[int, float]:
        """
        Run zero-shot prediction with LLaVA.
        """
        # Construct the prompt
        prompt = f"""<image>
This is a meme with the following text: "{text}"

Question: Is this meme hateful? Hateful memes contain hate speech, discrimination, or offensive content targeting specific groups based on race, religion, gender, etc.

Please answer with just "Yes" or "No"."""

        try:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                )
            
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Parse the textual response
            response_lower = response.lower()
            
            # Look for yes/no in the tail of the response
            if "yes" in response_lower[-50:]:
                return 1, 0.8
            elif "no" in response_lower[-50:]:
                return 0, 0.8
            else:
                return 0, 0.5
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.5


class SimpleVLMInference(ZeroShotInference):
    """
    Lightweight VLM inferencer (e.g., BLIP) for limited VRAM scenarios.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: torch.device = None,
    ):
        """
        Initialize the lightweight inferencer.

        Args:
            model_name: Hugging Face model identifier.
            device: torch.device override.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device)
        
        print(f"Loading BLIP model: {model_name}")
        
        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
            self.model.eval()
            
            print("✓ BLIP model ready")
            
        except Exception as e:
            print(f"✗ Failed to load BLIP: {e}")
            raise
    
    def predict(self, image: Image.Image, text: str) -> Tuple[int, float]:
        """
        Run zero-shot prediction using BLIP VQA.
        """
        question = f'This meme has the text "{text}". Is this meme hateful or offensive? Answer yes or no.'
        
        try:
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=10)
            
            answer = self.processor.decode(output[0], skip_special_tokens=True).lower().strip()
            
            if "yes" in answer:
                return 1, 0.7
            else:
                return 0, 0.7
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.5


def run_zero_shot_evaluation(
    model_type: str = "blip",
    output_path: str = "outputs/zero_shot_predictions.csv",
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run zero-shot evaluation.

    Args:
        model_type: One of {"qwen-vl", "llava", "blip"}.
        output_path: Where to save predictions.
        max_samples: Optional cap for quick experiments.

    Returns:
        Dict mapping metric name to value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build the requested inferencer
    if model_type == "qwen-vl":
        inference = QwenVLInference(device=device)
    elif model_type == "llava":
        inference = LLaVAInference(device=device)
    elif model_type == "blip":
        inference = SimpleVLMInference(device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load dataset
    print("\nLoading Hateful Memes test split...")
    dataset = load_dataset("neuralcatcher/hateful_memes")
    test_data = dataset["test"]
    
    if max_samples is not None:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    print(f"Test samples: {len(test_data)}")
    
    # Run inference
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print("\nRunning zero-shot inference...")
    for item in tqdm(test_data, desc="Predicting"):
        # Load image payload
        if "image" in item and item["image"] is not None:
            image = item["image"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
        else:
            continue
        
        text = item["text"]
        label = item["label"]
        
        # Predict
        pred, prob = inference.predict(image, text)
        
        all_labels.append(label)
        all_predictions.append(pred)
        all_probabilities.append(prob)
    
    # Compute metrics
    import numpy as np
    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    
    metrics = compute_metrics(labels, predictions, probabilities)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Zero-shot metrics ({model_type})")
    print("=" * 50)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  F1 (Macro):  {metrics['f1_macro']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    
    print_classification_report(labels, predictions)
    
    # Persist predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "prediction", "prob_hateful"])
        
        for i in range(len(labels)):
            writer.writerow([
                i,
                int(labels[i]),
                int(predictions[i]),
                f"{probabilities[i]:.6f}",
            ])
    
    print(f"\n✓ Predictions saved to: {output_path}")
    
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Large-model zero-shot Hateful Memes classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="blip",
        choices=["qwen-vl", "llava", "blip"],
        help="Model type: qwen-vl (large), llava (large), blip (compact).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/zero_shot_predictions.csv",
        help="Prediction CSV output path.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample cap for quick tests.",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("Hateful Memes zero-shot inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 60)
    
    metrics = run_zero_shot_evaluation(
        model_type=args.model,
        output_path=args.output,
        max_samples=args.max_samples,
    )

