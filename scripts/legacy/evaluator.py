"""
模型评估器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from utils.metrics import compute_metrics, get_predictions_and_probs

logger = logging.getLogger("hateful_memes")


class Evaluator:
    """
    模型评估器
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device
    ):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            device: 设备
        """
        self.model = model
        self.device = device
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_predictions: bool = False,
        output_path: Optional[str] = None
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            save_predictions: 是否保存预测结果
            output_path: 预测结果保存路径
            
        Returns:
            (metrics_dict, predictions_df) 元组
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        
        logger.info("开始评估...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                # 将数据移到设备上
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                
                # 记录 loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                
                # 获取预测和概率
                logits = outputs.logits
                preds, probs = get_predictions_and_probs(logits)
                
                all_predictions.extend(preds)
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probs)
        
        # 计算指标
        metrics = compute_metrics(
            predictions=np.array(all_predictions),
            labels=np.array(all_labels),
            probabilities=np.array(all_probabilities)
        )
        
        if total_loss > 0:
            metrics['loss'] = total_loss / len(dataloader)
        
        # 打印结果
        logger.info("\n评估结果:")
        logger.info("-" * 50)
        for key, value in metrics.items():
            logger.info(f"  {key.upper()}: {value:.4f}")
        logger.info("-" * 50)
        
        # 保存预测结果
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
                logger.info(f"✓ 预测结果已保存至: {output_path}")
        
        return metrics, predictions_df


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_predictions: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    便捷函数：评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        save_predictions: 是否保存预测结果
        output_path: 预测结果保存路径
        
    Returns:
        评估指标字典
    """
    evaluator = Evaluator(model, device)
    metrics, _ = evaluator.evaluate(
        dataloader,
        save_predictions=save_predictions,
        output_path=output_path
    )
    return metrics

