"""
快速测试脚本 - 验证数据加载和模型是否正常工作
"""

import torch
from utils import set_seed, setup_logger, get_device
from data import create_dataloader
from models import create_vilt_model, create_bert_model, create_vit_model


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*60)
    print("测试 1: 数据加载")
    print("="*60)
    
    # 测试 ViLT (多模态)
    print("\n1.1 测试 ViLT 数据加载...")
    _, processor = create_vilt_model(device=torch.device('cpu'))
    
    train_loader = create_dataloader(
        split='train',
        modality='multimodal',
        processor=processor,
        batch_size=2,
        num_workers=0,  # 测试时使用 0
        max_length=77
    )
    
    # 获取一个 batch
    batch = next(iter(train_loader))
    print(f"✓ ViLT batch keys: {batch.keys()}")
    print(f"  - pixel_values shape: {batch['pixel_values'].shape}")
    print(f"  - input_ids shape: {batch['input_ids'].shape}")
    print(f"  - labels shape: {batch['labels'].shape}")
    
    # 测试 BERT (文本)
    print("\n1.2 测试 BERT 数据加载...")
    _, tokenizer = create_bert_model(device=torch.device('cpu'))
    
    bert_loader = create_dataloader(
        split='validation',
        modality='text',
        processor=tokenizer,
        batch_size=2,
        num_workers=0
    )
    
    batch = next(iter(bert_loader))
    print(f"✓ BERT batch keys: {batch.keys()}")
    print(f"  - input_ids shape: {batch['input_ids'].shape}")
    
    # 测试 ViT (图像)
    print("\n1.3 测试 ViT 数据加载...")
    _, image_processor = create_vit_model(device=torch.device('cpu'))
    
    vit_loader = create_dataloader(
        split='validation',
        modality='image',
        processor=image_processor,
        batch_size=2,
        num_workers=0
    )
    
    batch = next(iter(vit_loader))
    print(f"✓ ViT batch keys: {batch.keys()}")
    print(f"  - pixel_values shape: {batch['pixel_values'].shape}")
    
    print("\n✅ 数据加载测试通过！")


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("测试 2: 模型前向传播")
    print("="*60)
    
    device = torch.device('cpu')
    
    # 测试 ViLT
    print("\n2.1 测试 ViLT 前向传播...")
    vilt_model, vilt_processor = create_vilt_model(device=device)
    
    vilt_loader = create_dataloader(
        split='validation',
        modality='multimodal',
        processor=vilt_processor,
        batch_size=2,
        num_workers=0
    )
    
    batch = next(iter(vilt_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = vilt_model(**batch)
    
    print(f"✓ ViLT 输出:")
    print(f"  - loss: {outputs.loss.item():.4f}")
    print(f"  - logits shape: {outputs.logits.shape}")
    
    # 测试 BERT
    print("\n2.2 测试 BERT 前向传播...")
    bert_model, bert_tokenizer = create_bert_model(device=device)
    
    bert_loader = create_dataloader(
        split='validation',
        modality='text',
        processor=bert_tokenizer,
        batch_size=2,
        num_workers=0
    )
    
    batch = next(iter(bert_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = bert_model(**batch)
    
    print(f"✓ BERT 输出:")
    print(f"  - loss: {outputs.loss.item():.4f}")
    print(f"  - logits shape: {outputs.logits.shape}")
    
    # 测试 ViT
    print("\n2.3 测试 ViT 前向传播...")
    vit_model, vit_processor = create_vit_model(device=device)
    
    vit_loader = create_dataloader(
        split='validation',
        modality='image',
        processor=vit_processor,
        batch_size=2,
        num_workers=0
    )
    
    batch = next(iter(vit_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = vit_model(**batch)
    
    print(f"✓ ViT 输出:")
    print(f"  - loss: {outputs.loss.item():.4f}")
    print(f"  - logits shape: {outputs.logits.shape}")
    
    print("\n✅ 模型前向传播测试通过！")


def main():
    """主函数"""
    # 设置日志和随机种子
    logger = setup_logger()
    set_seed(42)
    
    logger.info("\n" + "="*60)
    logger.info("开始快速测试...")
    logger.info("="*60)
    
    try:
        # 测试数据加载
        test_data_loading()
        
        # 测试模型前向传播
        test_model_forward()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！环境配置正确。")
        print("="*60)
        print("\n你可以开始训练模型了：")
        print("  python train_main.py --model_type vilt --batch_size 16 --num_epochs 2")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

