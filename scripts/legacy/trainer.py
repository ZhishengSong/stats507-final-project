"""
训练器模块
提供统一的训练循环和模型训练功能
"""

from typing import Dict, Optional, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from tqdm import tqdm

from eval.evaluator import Evaluator, compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import TrainingLogger


class Trainer:
    """
    统一训练器，支持 ViLT、BERT、ViT 三种模型的训练。
    
    功能：
    - 完整训练循环
    - 验证集评估
    - 早停机制
    - 检查点保存与恢复
    - 学习率调度
    
    Attributes:
        model: 待训练模型
        device: 计算设备
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        evaluator: 评估器
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: Optional[int] = None,
        scheduler_type: str = "linear",
    ):
        """
        初始化训练器。
        
        Args:
            model: 待训练模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_ratio: Warmup 比例
            total_steps: 总训练步数（用于调度器）
            scheduler_type: 调度器类型 ("linear", "cosine", "none")
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # 初始化优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 初始化学习率调度器
        self.scheduler = None
        if scheduler_type != "none" and total_steps is not None:
            warmup_steps = int(total_steps * warmup_ratio)
            
            if scheduler_type == "linear":
                self.scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
            elif scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,
                )
        
        # 初始化评估器
        self.evaluator = Evaluator(model, device)
        
        # 训练日志
        self.logger = TrainingLogger()
        
        # 最佳指标跟踪
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"✓ 训练器已初始化")
        print(f"  学习率: {learning_rate}")
        print(f"  权重衰减: {weight_decay}")
        print(f"  调度器: {scheduler_type}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        max_grad_norm: float = 1.0,
    ) -> float:
        """
        训练一个 epoch。
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch
            max_grad_norm: 梯度裁剪阈值
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}",
            leave=False,
        )
        
        for batch in progress_bar:
            # 将 batch 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        patience: int = 3,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
        metric_for_best: str = "auroc",
        max_grad_norm: float = 1.0,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        完整训练流程。
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练 epoch 数
            patience: 早停耐心值
            checkpoint_dir: 检查点保存目录
            model_name: 模型名称（用于保存文件）
            metric_for_best: 用于选择最佳模型的指标
            max_grad_norm: 梯度裁剪阈值
            resume_from: 恢复训练的检查点路径
        
        Returns:
            Dict: 训练历史和最佳指标
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        start_epoch = 1
        
        # 恢复训练
        if resume_from is not None:
            checkpoint_info = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device,
            )
            start_epoch = checkpoint_info["epoch"] + 1
            self.best_metric = checkpoint_info["metrics"].get(metric_for_best, 0.0)
            print(f"从 epoch {start_epoch} 继续训练")
        
        print(f"\n{'=' * 50}")
        print(f"开始训练")
        print(f"{'=' * 50}")
        print(f"  Epochs: {num_epochs}")
        print(f"  早停耐心值: {patience}")
        print(f"  最佳模型指标: {metric_for_best}")
        print(f"  检查点目录: {checkpoint_dir}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            # 训练一个 epoch
            train_loss = self.train_epoch(
                train_loader,
                epoch,
                max_grad_norm=max_grad_norm,
            )
            
            # 验证
            val_metrics, _, _, _ = self.evaluator.evaluate(
                val_loader,
                desc=f"Validating"
            )
            
            # 记录日志
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics.get("loss", 0.0),
                val_accuracy=val_metrics["accuracy"],
                val_auroc=val_metrics["auroc"],
            )
            
            # 检查是否为最佳模型
            current_metric = val_metrics[metric_for_best]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # 保存最佳模型
                best_path = checkpoint_dir / f"{model_name}_best.pt"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    str(best_path),
                    self.scheduler,
                )
                print(f"  ★ 新的最佳模型！{metric_for_best}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n早停触发！已 {patience} 个 epoch 未改善")
                    break
            
            # 保存最新检查点
            latest_path = checkpoint_dir / f"{model_name}_latest.pt"
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics,
                str(latest_path),
                self.scheduler,
            )
        
        # 保存训练历史
        self.logger.save_history(f"{model_name}_history.csv")
        
        print(f"\n{'=' * 50}")
        print(f"训练完成！")
        print(f"{'=' * 50}")
        print(f"  最佳 epoch: {self.best_epoch}")
        print(f"  最佳 {metric_for_best}: {self.best_metric:.4f}")
        
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "history": self.logger.history,
        }
    
    def load_best_model(self, checkpoint_path: str) -> None:
        """
        加载最佳模型检查点。
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        load_checkpoint(checkpoint_path, self.model, device=self.device)


if __name__ == "__main__":
    print("Trainer 模块测试")
    print("请通过 train.py 主脚本运行完整训练")

