"""
Hateful Memes 模型评估脚本

在测试集上评估已训练的模型，生成详细报告和预测结果。

使用示例:
    # 评估 ViLT 模型
    python evaluate.py --model vilt --checkpoint checkpoints/vilt_best.pt

    # 评估并保存预测结果
    python evaluate.py --model bert --checkpoint checkpoints/bert_best.pt --save_predictions
"""

import argparse

import torch

from data.dataset import get_dataloaders
from models.vilt import ViLTClassifier
from models.bert import BertClassifier
from models.vit import ViTClassifier
from eval.evaluator import Evaluator
from utils.seed import set_seed, get_device
from utils.checkpoint import load_model_only


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Hateful Memes 模型评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vilt", "bert", "vit"],
        help="模型类型",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="评估的数据集划分",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批次大小",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="文本最大长度",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader 工作进程数",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="是否保存预测结果",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    
    return parser.parse_args()


def create_model(model_type: str):
    """根据类型创建模型。"""
    if model_type == "vilt":
        return ViLTClassifier()
    elif model_type == "bert":
        return BertClassifier()
    elif model_type == "vit":
        return ViTClassifier()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def main():
    """主函数。"""
    args = parse_args()
    
    print("=" * 60)
    print("Hateful Memes 模型评估")
    print("=" * 60)
    print(f"模型类型: {args.model}")
    print(f"检查点: {args.checkpoint}")
    print(f"评估集: {args.split}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device()
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        model_type=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 选择对应的数据加载器
    if args.split == "train":
        dataloader = train_loader
    elif args.split == "validation":
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    # 创建并加载模型
    print("\n加载模型...")
    model = create_model(args.model)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"✓ 模型已加载: {args.checkpoint}")
    
    # 创建评估器
    evaluator = Evaluator(model, device)
    
    # 评估
    split_name = {
        "train": "Train",
        "validation": "Validation", 
        "test": "Test"
    }[args.split]
    
    metrics = evaluator.evaluate_and_report(dataloader, split_name=split_name)
    
    # 保存预测结果
    if args.save_predictions:
        from pathlib import Path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{args.model}_{args.split}_predictions.csv"
        evaluator.save_predictions(dataloader, str(output_path))
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()

