"""
交互式演示脚本 - 对单个图像和文本进行预测
"""

import torch
from PIL import Image
import argparse
from pathlib import Path

from utils import setup_logger, get_device, load_checkpoint
from models import create_vilt_model, create_bert_model, create_vit_model


def predict_single_sample(
    model,
    processor,
    image_path: str,
    text: str,
    device: torch.device,
    modality: str
):
    """
    预测单个样本
    
    Args:
        model: 模型
        processor: 处理器
        image_path: 图像路径
        text: 文本内容
        device: 设备
        modality: 模态类型
    """
    model.eval()
    
    # 加载图像
    if modality in ['multimodal', 'image']:
        image = Image.open(image_path).convert('RGB')
    
    # 预处理
    if modality == 'multimodal':
        inputs = processor(
            images=image,
            text=text,
            return_tensors='pt'
        )
    elif modality == 'text':
        inputs = processor(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
    elif modality == 'image':
        inputs = processor(
            images=image,
            return_tensors='pt'
        )
    
    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)
    
    # 返回结果
    label_map = {0: "Non-Hateful", 1: "Hateful"}
    
    return {
        'prediction': label_map[pred.item()],
        'confidence': probs[0, pred.item()].item(),
        'probabilities': {
            'non_hateful': probs[0, 0].item(),
            'hateful': probs[0, 1].item()
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="单样本预测演示")
    
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
        "--image_path",
        type=str,
        required=True,
        help="图像路径"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="文本内容"
    )
    parser.add_argument("--no_gpu", action="store_true", help="禁用 GPU")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    
    logger.info("="*60)
    logger.info("Hateful Memes 单样本预测演示")
    logger.info("="*60)
    
    # 检查文件是否存在
    if not Path(args.image_path).exists():
        logger.error(f"图像文件不存在: {args.image_path}")
        return
    
    if not Path(args.checkpoint_path).exists():
        logger.error(f"Checkpoint 文件不存在: {args.checkpoint_path}")
        return
    
    # 获取设备
    device = get_device(prefer_gpu=not args.no_gpu)
    
    # 确定模态
    modality_map = {
        "vilt": "multimodal",
        "bert": "text",
        "vit": "image"
    }
    modality = modality_map[args.model_type]
    
    # 加载模型
    logger.info(f"\n正在加载 {args.model_type.upper()} 模型...")
    
    if args.model_type == "vilt":
        model, processor = create_vilt_model(device=device)
    elif args.model_type == "bert":
        model, processor = create_bert_model(device=device)
    elif args.model_type == "vit":
        model, processor = create_vit_model(device=device)
    
    # 加载 checkpoint
    load_checkpoint(model, args.checkpoint_path, device=device)
    
    # 进行预测
    logger.info("\n进行预测...")
    logger.info(f"  图像: {args.image_path}")
    logger.info(f"  文本: {args.text}")
    
    result = predict_single_sample(
        model=model,
        processor=processor,
        image_path=args.image_path,
        text=args.text,
        device=device,
        modality=modality
    )
    
    # 打印结果
    logger.info("\n" + "="*60)
    logger.info("预测结果:")
    logger.info("="*60)
    logger.info(f"  预测标签: {result['prediction']}")
    logger.info(f"  置信度: {result['confidence']:.4f}")
    logger.info(f"\n  详细概率:")
    logger.info(f"    Non-Hateful: {result['probabilities']['non_hateful']:.4f}")
    logger.info(f"    Hateful: {result['probabilities']['hateful']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

