#!/usr/bin/env python3
"""
测试集图像质量诊断脚本（Top-K简化版）
- 基于Top-K置信度总和评估图像质量
- 解决Top-1置信度分散问题
- 移除易出错的语义簇匹配
"""

from defense_experiments import load_model, get_transform, get_mean_std, normalize_batch, load_image
from pathlib import Path
import torch

# ============= 类别关键词映射（仅用于显示，不参与计算） =============
CATEGORY_KEYWORDS = {
    "car": ["car", "vehicle", "车", "汽车", "轿车"],
    "dog": ["dog", "犬", "狗"],
    "bird": ["bird", "鸟", "雀"],
    "cat": ["cat", "猫"],
    "plane": ["plane", "aircraft", "飞机", "航空"],
    "ship": ["ship", "boat", "船", "舰"],
    "food": ["food", " dish", "餐", "食", "菜"],
    "furniture": ["furniture", "家具", "桌", "椅", "床"],
    "computer": ["computer", "pc", "电脑", "计算机"],
}


def load_imagenet_labels():
    """加载ImageNet类别标签"""
    try:
        with open("imagenet_classes.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except:
        return [f"class_{i}" for i in range(1000)]


def extract_display_category(filename: str) -> str:
    """
    从文件名提取显示用类别名（仅用于报告，不影响评分）
    """
    filename_lower = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in filename_lower:
                return category
    return "unknown"


def main():
    print("=" * 60)
    print("测试集图像质量诊断（Top-K简化版）")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 加载模型和预处理
    model = load_model(device, 'standard')
    transform = get_transform()
    mean, std = get_mean_std(device, torch.float32)

    # 加载ImageNet标签
    imagenet_labels = load_imagenet_labels()

    low_conf_images = []
    total_images = 0
    top_k = 5  # 使用Top-5置信度总和
    confidence_threshold = 0.7

    test_dir = Path('./test_set')
    if not test_dir.exists():
        print(f"❌ 错误: 目录 {test_dir} 不存在！")
        return

    print(f"诊断参数: Top-{top_k} ≥ {confidence_threshold}\n")
    print(
        f"{'图像名':<40s} {'Top-1预测':<18s} {'Top-1置信度':<12s} {'Top-{top_k}总和':<12s} {'推测类别':<10s} {'状态':<10s}")
    print("=" * 110)

    # 扫描所有jpg图像（包括子文件夹）
    for img_path in test_dir.rglob('*.jpg'):
        total_images += 1
        try:
            # 模型预测
            x = load_image(img_path, device, transform)
            logits = model(normalize_batch(x, mean, std))
            probs = torch.softmax(logits, dim=1)[0]

            # 获取Top-K结果
            topk_conf, topk_class = torch.topk(probs, top_k)

            top1_conf = topk_conf[0].item()
            top1_class = topk_class[0].item()
            topk_sum = topk_conf.sum().item()

            # 推测类别（仅显示用）
            guessed_category = extract_display_category(img_path.name)

            # 判断质量
            if topk_sum < confidence_threshold:
                low_conf_images.append((img_path.name, top1_conf, topk_sum, guessed_category))
                status = "❌ 低置信度"
            else:
                status = "✅"

            # 显示结果
            top1_label = imagenet_labels[top1_class][:16] if top1_class < len(
                imagenet_labels) else f"class_{top1_class}"
            print(
                f"{img_path.name:<40s} {top1_label:<18s} {top1_conf:<12.4f} {topk_sum:<12.4f} {guessed_category:<10s} {status:<10s}")

        except Exception as e:
            print(f"❌ {img_path.name:<40s} 加载失败: {e}")
            continue

    print("\n" + "=" * 110)
    print("诊断结果")
    print("=" * 110)

    if total_images == 0:
        print(f"⚠️  未找到任何jpg图像！")
        return

    low_conf_ratio = len(low_conf_images) / total_images

    print(f"总图像数: {total_images}")
    print(f"高置信度图像(Top-{top_k} ≥ {confidence_threshold}): {total_images - len(low_conf_images)}")
    print(f"低置信度图像: {len(low_conf_images)}")
    print(f"低置信度比例: {low_conf_ratio:.1%}")

    if low_conf_ratio > 0.3:
        print(f"\n⚠️  警告: 超过30%的图像置信度不足！")
        print("   建议重新筛选测试集！")
        print(f"\n低置信度图像列表:")
        for name, top1_conf, topk_sum, category in low_conf_images:
            print(f"   - {name}: Top-1={top1_conf:.4f}, Top-{top_k}总和={topk_sum:.4f} ({category})")
    else:
        print("\n✅ 测试集质量合格！")
        if len(low_conf_images) > 0:
            print(f"   但建议检查以下低置信度图像:")
            for name, top1_conf, topk_sum, category in low_conf_images:
                print(f"   - {name}: Top-1={top1_conf:.4f}, Top-{top_k}总和={topk_sum:.4f} ({category})")


if __name__ == "__main__":
    main()