import argparse
import io
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from defense_experiments import run_attack, get_mean_std, get_transform


# ============ 从原有代码保留的函数 ============
# 请确保这些函数已在你的文件中定义
# load_model, get_transform, load_image, get_mean_std, predict,
# run_attack, normalize_batch, defend_input, is_adversarial_by_feature
# 如果缺少，请复制之前的实现

# ============ 1. 扩展模型加载函数 ============
def load_model(device: torch.device, model_type: str = "resnet50") -> torch.nn.Module:
    """支持多种架构的模型加载"""
    try:
        if model_type == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_type == "vgg19":
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        elif model_type == "densenet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        elif model_type == "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        elif model_type == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        print(f"Warning: Could not load pretrained weights for {model_type}: {e}")
        # 回退到旧版API
        if model_type == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_type == "vgg19":
            model = models.vgg19(pretrained=True)
        elif model_type == "densenet121":
            model = models.densenet121(pretrained=True)
        else:
            raise ValueError(f"Model {model_type} not available")

    return model.eval().to(device)


# ============ 2. 迁移攻击核心函数 ============
def evaluate_transfer_attack(
        source_model: torch.nn.Module,
        target_models: Dict[str, torch.nn.Module],
        x: torch.Tensor,
        y_true: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        attack_name: str,
        eps: float,
        alpha: float,
        steps: int,
        cw_c: float,
        cw_kappa: float,
        cw_steps: int,
        cw_lr: float,
) -> Dict[str, Dict]:
    """
    评估对抗样本从源模型到目标模型的迁移性

    Returns:
        每个目标模型的攻击结果统计
    """
    results = {}

    # 1. 在源模型上生成对抗样本（白盒攻击）
    x_adv = run_attack(
        attack_name,
        source_model,
        x,
        y_true,
        mean,
        std,
        eps,
        alpha,
        steps,
        cw_c,
        cw_kappa,
        cw_steps,
        cw_lr,
    )

    # 2. 在源模型上验证攻击效果（基线）
    logits_source = predict(source_model, x_adv, mean, std)
    pred_source = logits_source.argmax(dim=1)
    attack_success_source = (pred_source != y_true).item()

    results["source_model"] = {
        "model_name": "ResNet50",
        "attack_success": attack_success_source,
        "predicted_label": pred_source.item(),
        "clean_label": y_true.item(),
    }

    # 3. 在目标模型上测试迁移性（黑盒攻击）
    for model_name, target_model in target_models.items():
        with torch.no_grad():
            # 注意：目标模型可能使用不同的mean/std，但为公平对比，统一使用源模型的归一化
            logits_target = predict(target_model, x_adv, mean, std)
            pred_target = logits_target.argmax(dim=1)
            attack_success_target = (pred_target != y_true).item()

        results[model_name] = {
            "model_name": model_name,
            "attack_success": attack_success_target,
            "predicted_label": pred_target.item(),
            "clean_label": y_true.item(),
            "transfer_success": attack_success_target,  # 是否迁移成功
        }

    return results, x_adv


# ============ 3. 主实验流程 ============
def main():
    parser = argparse.ArgumentParser(description="Black-box Transferability Attack")

    # 模型选择
    parser.add_argument(
        "--source_model",
        type=str,
        default="resnet50",
        choices=["resnet50", "vgg19", "densenet121", "vit_b_16"],
        help="Source model to generate adversarial examples (white-box)"
    )
    parser.add_argument(
        "--target_models",
        type=str,
        nargs="+",
        default=["vgg19", "densenet121", "vit_b_16"],
        choices=["resnet50", "vgg19", "densenet121", "vit_b_16",
                 "efficientnet_b0", "mobilenet_v2", "regnet_y_400mf"],
        help="Target models to evaluate transferability (black-box)"
    )

    # 数据输入
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="example.jpg",
        help="Single test image path"
    )

    # 攻击参数
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["pgd"],
        choices=["fgsm", "pgd", "cw"],
        help="Attack methods to test"
    )
    parser.add_argument(
        "--eps_list",
        type=float,
        nargs="+",
        default=[4 / 255, 8 / 255, 16 / 255],
        help="Perturbation budgets (Linf)"
    )

    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cw_c", type=float, default=1.0)
    parser.add_argument("--cw_kappa", type=float, default=0.0)
    parser.add_argument("--cw_steps", type=int, default=100)
    parser.add_argument("--cw_lr", type=float, default=0.01)

    # 输出设置
    parser.add_argument(
        "--save_adv_images",
        action="store_true",
        help="Save generated adversarial images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./transfer_results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # ============ 实验准备 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载源模型（白盒）
    print(f"\n[1/4] Loading source model: {args.source_model}")
    source_model = load_model(device, args.source_model)

    # 加载目标模型（黑盒）
    print(f"\n[2/4] Loading target models: {args.target_models}")
    target_models = {}
    for model_name in args.target_models:
        if model_name != args.source_model:
            target_models[model_name] = load_model(device, model_name)
        else:
            print(f"  Skipping {model_name} (same as source model)")

    # 准备数据
    transform = get_transform()
    mean, std = get_mean_std(device, torch.float32)

    # 获取图片列表
    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        print(f"\n[3/4] Found {len(image_paths)} images in {image_dir}")
    else:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_paths = [image_path]
        print(f"\n[3/4] Using single image: {image_path}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============ 实验执行 ============
    print("\n[4/4] Running transfer attack experiments...")
    print("=" * 80)

    # 结果统计结构
    all_results = {}
    for attack_name in args.attacks:
        all_results[attack_name] = {}
        for eps in args.eps_list:
            all_results[attack_name][eps] = {
                "source_success": [],
                "transfer_success": {name: [] for name in target_models.keys()}
            }

    # 遍历图片、攻击方法、扰动强度
    for img_idx, image_path in enumerate(image_paths):
        print(f"\n[Image {img_idx + 1}/{len(image_paths)}] {image_path.name}")

        # 加载图像
        x = load_image(image_path, device, transform)

        # 获取真实标签（使用源模型预测作为伪标签）
        with torch.no_grad():
            logits_clean = predict(source_model, x, mean, std)
            y_true = logits_clean.argmax(dim=1)
            clean_label = y_true.item()

        print(f"  Clean label: {clean_label}")

        for attack_name in args.attacks:
            for eps in args.eps_list:
                print(f"  Running {attack_name.upper()} with eps={eps:.5f}...")

                # 执行迁移攻击评估
                results, x_adv = evaluate_transfer_attack(
                    source_model=source_model,
                    target_models=target_models,
                    x=x,
                    y_true=y_true,
                    mean=mean,
                    std=std,
                    attack_name=attack_name,
                    eps=eps,
                    alpha=args.alpha,
                    steps=args.steps,
                    cw_c=args.cw_c,
                    cw_kappa=args.cw_kappa,
                    cw_steps=args.cw_steps,
                    cw_lr=args.cw_lr,
                )

                # 记录结果
                source_success = results["source_model"]["attack_success"]
                all_results[attack_name][eps]["source_success"].append(source_success)

                print(f"    Source model success: {source_success}")

                for model_name in target_models.keys():
                    transfer_success = results[model_name]["transfer_success"]
                    all_results[attack_name][eps]["transfer_success"][model_name].append(transfer_success)
                    print(f"    Transfer to {model_name}: {transfer_success}")

                # 保存对抗样本图片（可选）
                if args.save_adv_images:
                    adv_img_dir = output_dir / f"{attack_name}_eps_{eps:.5f}"
                    adv_img_dir.mkdir(exist_ok=True)
                    adv_img_path = adv_img_dir / f"adv_{image_path.stem}.png"
                    # 反归一化并保存
                    x_adv_denorm = x_adv * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
                    x_adv_denorm = torch.clamp(x_adv_denorm, 0, 1)
                    save_image(x_adv_denorm.squeeze(0), adv_img_path)

    # ============ 结果汇总与统计 ============
    print("\n" + "=" * 80)
    print("TRANSFERABILITY SUMMARY")
    print("=" * 80)

    # 打印表格头
    header = f"{'Attack':<10} {'Eps':<10} {'Source':<10}"
    for model_name in target_models.keys():
        header += f" {model_name:<15}"
    print(header)
    print("-" * len(header))

    # 打印每个配置的统计结果
    summary_data = []
    for attack_name in args.attacks:
        for eps in args.eps_list:
            # 计算平均成功率
            source_success_rate = sum(all_results[attack_name][eps]["source_success"]) / len(image_paths)

            row = f"{attack_name:<10} {eps:<10.5f} {source_success_rate:<10.3f}"

            transfer_rates = {}
            for model_name in target_models.keys():
                successes = all_results[attack_name][eps]["transfer_success"][model_name]
                transfer_rate = sum(successes) / len(image_paths)
                transfer_rates[model_name] = transfer_rate
                row += f" {transfer_rate:<15.3f}"

            print(row)
            summary_data.append({
                "attack": attack_name,
                "eps": eps,
                "source_rate": source_success_rate,
                "transfer_rates": transfer_rates
            })

    # ============ 保存详细结果 ============
    results_file = output_dir / "transfer_results.json"
    import json
    # 转换numpy数据类型以支持JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    print(f"\nDetailed results saved to: {results_file}")

    # ============ 生成迁移性热力图（可选）===========
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 创建迁移性矩阵
        model_names = list(target_models.keys())
        eps_values = args.eps_list

        for attack_name in args.attacks:
            matrix = np.zeros((len(eps_values), len(model_names)))
            for i, eps in enumerate(eps_values):
                for j, model_name in enumerate(model_names):
                    rates = all_results[attack_name][eps]["transfer_success"][model_name]
                    matrix[i, j] = sum(rates) / len(rates)

            # 绘制热力图
            plt.figure(figsize=(10, 6))
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                        xticklabels=model_names,
                        yticklabels=[f"{e:.3f}" for e in eps_values])
            plt.title(
                f'Transferability Attack Success Rates\nSource: {args.source_model}, Attack: {attack_name.upper()}')
            plt.xlabel('Target Models (Black-box)')
            plt.ylabel('Perturbation Budget (eps)')
            plt.tight_layout()

            plot_path = output_dir / f"transfer_heatmap_{attack_name}.png"
            plt.savefig(plot_path, dpi=300)
            print(f"Transferability heatmap saved: {plot_path}")

    except ImportError:
        print("\nInstall matplotlib and seaborn for visualization: pip install matplotlib seaborn")


# ============ 辅助函数：保存图像 ============
def save_image(tensor: torch.Tensor, path: Path):
    """保存张量为图像文件"""
    from torchvision.utils import save_image
    save_image(tensor, str(path))


# ============ 4. 主入口 ============
if __name__ == "__main__":
    main()