"""
Zero-shot inferencer that supports Qwen-VL, LLaVA, BLIP-2, and similar LMMs.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal
import logging

from utils.metrics import compute_metrics

logger = logging.getLogger("hateful_memes")


class ZeroShotInferencer:
    """
    Wrapper around large multimodal models to run zero-shot inference.
    """
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        model_type: Literal['qwen-vl', 'llava', 'blip2'] = 'qwen-vl'
    ):
        """
        Initialize the inferencer.

        Args:
            model_name: Model identifier or path.
            device: torch.device to run on.
            model_type: One of the supported model families.
        """
        self.model_name = model_name
        self.device = device
        self.model_type = model_type
        
        logger.info(f"Loading {model_type} model: {model_name}")
        
        try:
            if model_type == 'qwen-vl':
                self._load_qwen_vl()
            elif model_type == 'llava':
                self._load_llava()
            elif model_type == 'blip2':
                self._load_blip2()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info("✓ Zero-shot model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load zero-shot model: {e}")
            raise
    
    def _load_qwen_vl(self):
        """Load the Qwen-VL model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        ).eval()
        
        self.prompt_template = """Please analyze this meme image with text: "{text}"

Question: Is this meme hateful or offensive?
Answer with ONLY one word: Yes or No."""
    
    def _load_llava(self):
        """Load LLaVA (not yet implemented)."""
        logger.warning("LLaVA loading placeholder; please use the official llava package.")
        raise NotImplementedError("LLaVA support is pending.")
    
    def _load_blip2(self):
        """Load BLIP-2 as a fallback option."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        ).eval()
        
        self.prompt_template = """Question: Is this meme with text "{text}" hateful or offensive? Answer Yes or No."""
    
    def predict_single(
        self,
        image,
        text: str
    ) -> Tuple[int, float]:
        """
        Run inference for a single (image, text) pair.

        Args:
            image: PIL Image.
            text: Associated meme caption.

        Returns:
            Tuple of (prediction, probability).
        """
        prompt = self.prompt_template.format(text=text)
        
        if self.model_type == 'qwen-vl':
            return self._predict_qwen_vl(image, prompt)
        elif self.model_type == 'blip2':
            return self._predict_blip2(image, prompt)
        else:
            raise NotImplementedError(f"{self.model_type} prediction support is not implemented yet.")
    
    def _predict_qwen_vl(self, image, prompt: str) -> Tuple[int, float]:
        """Predict using Qwen-VL."""
        # Build the multi-modal query
        query = self.tokenizer.from_list_format([
            {'image': image},  # PIL Image
            {'text': prompt},
        ])
        
        # Generate the answer
        with torch.no_grad():
            response, _ = self.model.chat(
                self.tokenizer,
                query=query,
                history=None
            )
        
        # Interpret the answer
        response_lower = response.lower().strip()
        
        # Use simple keyword heuristics
        if 'yes' in response_lower or 'hateful' in response_lower or 'offensive' in response_lower:
            prediction = 1
            probability = 0.9  # confident positive
        elif 'no' in response_lower or 'not' in response_lower:
            prediction = 0
            probability = 0.1  # confident negative
        else:
            # Model response is ambiguous
            prediction = 0
            probability = 0.5
        
        return prediction, probability
    
    def _predict_blip2(self, image, prompt: str) -> Tuple[int, float]:
        """Predict using BLIP-2."""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        response_lower = response.lower()
        
        # Interpret the decoded text
        if 'yes' in response_lower or 'hateful' in response_lower:
            prediction = 1
            probability = 0.9
        elif 'no' in response_lower or 'not' in response_lower:
            prediction = 0
            probability = 0.1
        else:
            prediction = 0
            probability = 0.5
        
        return prediction, probability
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_predictions: bool = False,
        output_path: Optional[str] = None
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        """
        Run zero-shot evaluation over a dataloader.

        Args:
            dataloader: Must return raw `image` and `text` entries.
            save_predictions: Whether to persist predictions.
            output_path: Optional CSV path.

        Returns:
            Tuple of (metrics_dict, predictions_df).
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        logger.info("Starting zero-shot inference...")
        
        for batch in tqdm(dataloader, desc="Zero-shot inference"):
            # Dataloader must provide raw image/text pairs
            images = batch['image'] if 'image' in batch else None
            texts = batch['text'] if 'text' in batch else None
            labels = batch['labels']
            
            if images is None or texts is None:
                logger.error("Dataloader must yield 'image' and 'text' fields.")
                raise ValueError("Raw image and text data are required.")
            
            # Predict sample-by-sample (models do not batch well here)
            for image, text, label in zip(images, texts, labels):
                pred, prob = self.predict_single(image, text)
                all_predictions.append(pred)
                all_probabilities.append(prob)
                all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        
        # Compute metrics
        metrics = compute_metrics(
            predictions=np.array(all_predictions),
            labels=np.array(all_labels),
            probabilities=np.array(all_probabilities)
        )
        
        # Log the summary
        logger.info("\nZero-shot metrics:")
        logger.info("-" * 50)
        for key, value in metrics.items():
            logger.info(f"  {key.upper()}: {value:.4f}")
        logger.info("-" * 50)
        
        # Optionally save predictions
        predictions_df = None
        if save_predictions:
            predictions_df = pd.DataFrame({
                'label': all_labels,
                'prediction': all_predictions,
                'probability': all_probabilities
            })
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                predictions_df.to_csv(output_path, index=False)
                logger.info(f"✓ Predictions saved to {output_path}")
        
        return metrics, predictions_df

