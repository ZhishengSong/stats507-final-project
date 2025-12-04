"""
全局配置文件
"""

from pathlib import Path
from typing import Dict, Any


class Config:
    """项目配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent
    
    # 数据相关
    DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
    
    # 模型相关
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    
    # 预训练模型名称
    PRETRAINED_MODELS = {
        'vilt': 'dandelin/vilt-b32-mlm',
        'bert': 'bert-base-uncased',
        'vit': 'google/vit-base-patch16-224',
        'qwen-vl': 'Qwen/Qwen-VL-Chat',
        'llava': 'llava-hf/llava-1.5-7b-hf',
        'blip2': 'Salesforce/blip2-opt-2.7b'
    }
    
    # 训练超参数（默认值）
    TRAINING_CONFIG = {
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 5e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'accumulation_steps': 1,
        'early_stopping_patience': 3,
        'metric_for_best_model': 'auroc'
    }
    
    # 数据处理配置
    DATA_CONFIG = {
        'max_length': 77,
        'num_workers': 4,
    }
    
    # 随机种子
    SEED = 42
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """
        获取特定模型的配置
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型配置字典
        """
        config = {
            'pretrained_model': cls.PRETRAINED_MODELS.get(model_type),
            'save_dir': cls.CHECKPOINT_DIR / model_type,
            **cls.TRAINING_CONFIG,
            **cls.DATA_CONFIG
        }
        return config
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        cls.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        for model_type in ['vilt', 'bert', 'vit']:
            (cls.CHECKPOINT_DIR / model_type).mkdir(parents=True, exist_ok=True)


# 初始化配置（创建必要的目录）
Config.create_directories()

