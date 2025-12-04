"""
主评估脚本 - 在测试集上评估训练好的模型
"""

import argparse
import torch
from pathlib import Path

from utils import set_seed, setup_logger, get_device, load_checkpoint
from data import create_dataloader
from models import create_vilt_model, create_bert_model, create_vit_model
from eval import evaluate_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估 Hateful Memes 分类模型")
    
    # 模型相关
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["vilt", "bert", "vit"],
        help="模型类型"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="模型 checkpoint 路径"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="预训练模型名称（默认：根据 model_type 自动选择）"
    )
    
    # 数据相关
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="评估的数据集划分"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--max_length", type=int, default=77, help="文本最大长度")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载的工作进程数")
    parser.add_argument("--cache_dir", type=str, default=None, help="数据集缓存目录")
    
    # 输出相关
    parser.add_argument("--save_predictions", action="store_true", help="保存预测结果")
    parser.add_argument("--output_path", type=str, default=None, help="预测结果保存路径")
    
    # 其他
    parser.add_argument("--log_file", type=str, default=None, help="日志文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_gpu", action="store_true", help="禁用 GPU")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logger = setup_logger(log_file=args.log_file)
    logger.info("=" * 80)
    logger.info("Hateful Memes 模型评估")
    logger.info("=" * 80)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device(prefer_gpu=not args.no_gpu)
    
    # 确定模型名称
    model_name_map = {
        "vilt": "dandelin/vilt-b32-mlm",
        "bert": "bert-base-uncased",
        "vit": "google/vit-base-patch16-224"
    }
    pretrained_model = args.pretrained_model or model_name_map[args.model_type]
    
    # 确定模态类型
    modality_map = {
        "vilt": "multimodal",
        "bert": "text",
        "vit": "image"
    }
    modality = modality_map[args.model_type]
    
    logger.info(f"\n配置信息:")
    logger.info(f"  模型类型: {args.model_type}")
    logger.info(f"  Checkpoint: {args.checkpoint_path}")
    logger.info(f"  评估集: {args.split}")
    logger.info(f"  批次大小: {args.batch_size}")
    
    # 创建模型和处理器
    logger.info(f"\n正在加载 {args.model_type.upper()} 模型...")
    
    if args.model_type == "vilt":
        model, processor = create_vilt_model(
            pretrained_model_name=pretrained_model,
            device=device
        )
    elif args.model_type == "bert":
        model, processor = create_bert_model(
            pretrained_model_name=pretrained_model,
            device=device
        )
    elif args.model_type == "vit":
        model, processor = create_vit_model(
            pretrained_model_name=pretrained_model,
            device=device
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 加载 checkpoint
    logger.info(f"正在加载 checkpoint: {args.checkpoint_path}")
    checkpoint_info = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint_path,
        device=device
    )
    logger.info(f"  Checkpoint epoch: {checkpoint_info['epoch']}")
    
    # 创建数据加载器
    logger.info(f"\n正在加载 {args.split} 数据集...")
    
    dataloader = create_dataloader(
        split=args.split,
        modality=modality,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        shuffle=False
    )
    
    # 确定输出路径
    output_path = args.output_path
    if args.save_predictions and output_path is None:
        checkpoint_dir = Path(args.checkpoint_path).parent
        output_path = checkpoint_dir / f"predictions_{args.split}.csv"
    
    # 评估
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        save_predictions=args.save_predictions,
        output_path=output_path
    )
    
    logger.info("\n评估完成!")


if __name__ == "__main__":
    main()

