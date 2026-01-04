#!/usr/bin/env python3
"""
对抗攻击可视化分析器
支持 FGSM / PGD / CW-L2 攻击方法的全面评估
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image


from dataclasses import dataclass
from typing import Optional
# ==================== 你的攻击代码（保持不变） ====================



def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("mean/std must be 1D tensors with shape [C]")
    if x.ndim != 4:
        raise ValueError("x must be a 4D tensor with shape [N, C, H, W]")
    mean = mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def fgsm_attack(
        model: torch.nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        eps: float,
        mean: torch.Tensor,
        std: torch.Tensor,
) -> torch.Tensor:
    x_in = x.detach().clone().requires_grad_(True)
    logits = model(normalize_batch(x_in, mean, std))
    loss = F.cross_entropy(logits, y_true)
    grad = torch.autograd.grad(loss, x_in, only_inputs=True)[0]
    x_adv = x_in + eps * grad.sign()
    return x_adv.clamp(0.0, 1.0).detach()


def pgd_linf_attack(
        model: torch.nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        eps: float,
        alpha: float,
        steps: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        random_start: bool = True,
) -> torch.Tensor:
    x_orig = x.detach()
    if random_start:
        x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x_orig.clone()

    for _ in range(int(steps)):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        logits = model(normalize_batch(x_adv, mean, std))
        loss = F.cross_entropy(logits, y_true)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()


def _atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass(frozen=True)
class CWResult:
    x_adv: torch.Tensor
    success: torch.Tensor


def cw_l2_attack(
        model: torch.nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        c: float = 1.0,
        kappa: float = 0.0,
        steps: int = 1000,
        lr: float = 1e-2,
        targeted: bool = False,
        y_target: Optional[torch.Tensor] = None,
) -> CWResult:
    if targeted and y_target is None:
        raise ValueError("y_target must be provided when targeted=True")

    x0 = x.detach().clamp(0.0, 1.0)
    eps = 1e-6
    x0_tanh = x0 * (1.0 - 2.0 * eps) + eps
    w0 = _atanh(x0_tanh * 2.0 - 1.0).detach()
    w = w0.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = x0.clone()
    best_l2 = torch.full((x0.shape[0],), float("inf"), device=x0.device, dtype=x0.dtype)
    best_success = torch.zeros((x0.shape[0],), device=x0.device, dtype=torch.bool)

    y_cmp = y_target if targeted else y_true

    for _ in range(int(steps)):
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = model(normalize_batch(x_adv, mean, std))

        num_classes = logits.shape[1]
        y_onehot = F.one_hot(y_cmp, num_classes=num_classes).to(dtype=logits.dtype)

        real = (logits * y_onehot).sum(dim=1)
        other = (logits - 1e4 * y_onehot).amax(dim=1)

        if targeted:
            f = torch.clamp(other - real + kappa, min=0.0)
            success = (logits.argmax(dim=1) == y_cmp)
        else:
            f = torch.clamp(real - other + kappa, min=0.0)
            success = (logits.argmax(dim=1) != y_cmp)

        l2 = (x_adv - x0).view(x0.shape[0], -1).pow(2).sum(dim=1)
        loss = (l2 + c * f).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        improved = success & (l2 < best_l2)
        if improved.any():
            best_l2 = torch.where(improved, l2, best_l2)
            best_success = best_success | improved
            best_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv.detach(), best_adv)

    final_adv = torch.where(best_success.view(-1, 1, 1, 1), best_adv, (0.5 * (torch.tanh(w) + 1.0)).detach())
    return CWResult(x_adv=final_adv, success=best_success)


# ==================== 可视化核心类 ====================
class AttackVisualizer:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 mean: torch.Tensor, std: torch.Tensor,
                 imagenet_classes: list = None):
        self.model = model
        self.device = device
        self.mean = mean
        self.std = std
        self.imagenet_classes = imagenet_classes or self._load_imagenet_classes()

    def _load_imagenet_classes(self) -> list:
        """加载ImageNet类别标签"""
        import requests
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.text.strip().split('\n')
        except:
            pass
        # 备用：生成通用标签
        return [f"class_{i}" for i in range(1000)]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化到[0,1]"""
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    def predict(self, x: torch.Tensor) -> Tuple[int, str, float]:
        """预测并返回(类别ID, 名称, 置信度)"""
        with torch.no_grad():
            logits = self.model(normalize_batch(x, self.mean, self.std))
            probs = F.softmax(logits, dim=1)
            pred_id = probs.argmax(dim=1).item()
            confidence = probs[0, pred_id].item()
            class_name = self.imagenet_classes[pred_id]
        return pred_id, class_name, confidence

    def calculate_metrics(self, x_clean: torch.Tensor, x_adv: torch.Tensor) -> Dict:
        """计算攻击的定量指标"""
        diff = x_adv - x_clean

        # Lp范数
        l_inf = torch.max(torch.abs(diff)).item()
        l_2 = torch.norm(diff, p=2).item()
        l_1 = torch.norm(diff, p=1).item()

        # 视觉相似度
        x_clean_np = x_clean.cpu().numpy()
        x_adv_np = x_adv.cpu().numpy()

        # SSIM (结构相似性)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(
                x_clean_np[0].transpose(1, 2, 0),
                x_adv_np[0].transpose(1, 2, 0),
                multichannel=True,
                data_range=1.0
            )
        except ImportError:
            ssim_score = 0.0

        # PSNR (峰值信噪比)
        mse = torch.mean(diff ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        # 高频扰动比例
        fft_diff = torch.fft.fft2(diff)
        high_freq_ratio = (torch.abs(fft_diff) > 0.1).float().mean().item()

        # 扰动像素比例
        perturbed_pixels = (torch.abs(diff) > 1 / 255).float().mean().item()

        return {
            "L∞ (pixel)": l_inf,
            "L2": l_2,
            "L1": l_1,
            "SSIM": ssim_score,
            "PSNR": psnr,
            "Perturbed Pixels %": perturbed_pixels * 100,
            "High Freq Ratio": high_freq_ratio * 100,
        }

    def visualize_attack_grid(self, x_clean: torch.Tensor,
                              results: Dict[str, Dict],
                              save_path: Path = None):
        """
        生成攻击效果图网格：原图 + 对抗样本 + 扰动放大
        """
        n_attacks = len(results)
        fig = plt.figure(figsize=(4 * 3, 4 * n_attacks))

        # 准备数据
        x_clean_denorm = self.denormalize(x_clean).squeeze()

        for idx, (attack_name, result) in enumerate(results.items()):
            x_adv = result["x_adv"]
            x_adv_denorm = self.denormalize(x_adv).squeeze()
            diff = (x_adv - x_clean).squeeze()

            # Row 1: 原图 vs 对抗样本
            ax1 = plt.subplot(n_attacks, 3, idx * 3 + 1)
            self._plot_image_pair(ax1, x_clean_denorm, x_adv_denorm,
                                  "Original", "Adversarial")

            # Row 2: 扰动图 ×10
            ax2 = plt.subplot(n_attacks, 3, idx * 3 + 2)
            diff_10x = torch.clamp(x_clean_denorm + 10 * diff, 0, 1)
            self._plot_image(ax2, diff_10x, "Perturbation ×10")

            # Row 3: 扰动图 ×50
            ax3 = plt.subplot(n_attacks, 3, idx * 3 + 3)
            diff_50x = torch.clamp(x_clean_denorm + 50 * diff, 0, 1)
            self._plot_image(ax3, diff_50x, "Perturbation ×50")

            # 添加攻击信息
            pred_clean = result["pred_clean"]
            pred_adv = result["pred_adv"]
            success = "✅ SUCCESS" if pred_clean[0] != pred_adv[0] else "❌ FAILED"
            fig.text(0.5, 1 - (idx * (1 / n_attacks) - 0.02),
                     f"{attack_name.upper()} Attack - {success}",
                     ha='center', va='top', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_image(self, ax, img_tensor, title):
        """绘制单张图像"""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    def _plot_image_pair(self, ax, img1, img2, title1, title2):
        """绘制对比图像"""
        img_np = torch.cat([img1, img2], dim=2).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(f"{title1} vs {title2}", fontsize=11)
        ax.axis('off')
        # 添加分割线
        h, w = img1.shape[1:]
        ax.axvline(x=w, color='white', linewidth=2)

    def visualize_attack_trajectory(self, x_clean: torch.Tensor, y_true: int,
                                    attack_name: str, eps: float, alpha: float, steps: int,
                                    save_path: Path = None):
        """
        动态展示攻击过程：置信度变化和扰动增长
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 存储轨迹数据
        traj_probs = []
        traj_perturbs = []

        # 执行攻击并记录中间结果
        x_orig = x_clean.detach()
        x_adv = x_orig.clone()
        if attack_name == "pgd":
            x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0, 1)

        for step in range(steps + 1):
            if step > 0:
                if attack_name == "pgd":
                    # PGD单步
                    x_adv = x_adv.detach().clone().requires_grad_(True)
                    logits = self.model(normalize_batch(x_adv, self.mean, self.std))
                    loss = F.cross_entropy(logits, torch.tensor([y_true], device=self.device))
                    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
                    x_adv = x_adv + alpha * grad.sign()
                    x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0, 1)
                elif attack_name == "fgsm":
                    # FGSM一次性
                    if step == 1:
                        x_in = x_orig.clone().requires_grad_(True)
                        logits = self.model(normalize_batch(x_in, self.mean, self.std))
                        loss = F.cross_entropy(logits, torch.tensor([y_true], device=self.device))
                        grad = torch.autograd.grad(loss, x_in, only_inputs=True)[0]
                        x_adv = x_orig + eps * grad.sign()
                        x_adv = x_adv.clamp(0, 1)
                    break

            # 记录当前状态
            with torch.no_grad():
                logits = self.model(normalize_batch(x_adv, self.mean, self.std))
                probs = F.softmax(logits, dim=1)
                traj_probs.append(probs[0, [y_true, 805]].cpu().numpy())  # 熊猫和足球
                traj_perturbs.append(torch.norm(x_adv - x_orig).item())

        traj_probs = np.array(traj_probs)

        # 绘制概率轨迹
        ax1.plot(traj_probs[:, 0], label='Original Class (Panda)', color='green', linewidth=2, marker='o', markersize=3)
        ax1.plot(traj_probs[:, 1], label='Target Class (Soccer)', color='red', linewidth=2, marker='x', markersize=3)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')
        ax1.set_xlabel('Attack Step', fontsize=12)
        ax1.set_ylabel('Prediction Probability', fontsize=12)
        ax1.set_title(f'{attack_name.upper()} Attack Trajectory (Eps={eps:.3f})', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # 绘制扰动增长
        ax2.plot(traj_perturbs, color='purple', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Attack Step', fontsize=12)
        ax2.set_ylabel('L2 Perturbation', fontsize=12)
        ax2.set_title('Perturbation Growth', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def visualize_perturbation_analysis(self, x_clean: torch.Tensor,
                                        results: Dict[str, Dict],
                                        save_path: Path = None):
        """
        扰动的频域和空域分析
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Perturbation Spatial & Frequency Analysis', fontsize=16, fontweight='bold')

        # 1. 空域分布直方图
        for idx, (attack_name, result) in enumerate(results.items()):
            diff = (result["x_adv"] - x_clean).squeeze().cpu().numpy()
            axes[0, idx].hist(diff.flatten(), bins=50, alpha=0.7, color=['red', 'blue', 'purple'][idx])
            axes[0, idx].set_title(f'{attack_name.upper()} Perturbation Distribution', fontsize=12)
            axes[0, idx].set_xlabel('Perturbation Value')
            axes[0, idx].set_ylabel('Frequency')
            axes[0, idx].grid(True, alpha=0.3)

        # 2. 频域分析
        for idx, (attack_name, result) in enumerate(results.items()):
            diff = (result["x_adv"] - x_clean).squeeze().cpu().numpy()
            fft_diff = np.fft.fft2(diff.transpose(1, 2, 0).mean(axis=2))
            fft_magnitude = np.abs(np.fft.fftshift(fft_diff))

            im = axes[1, idx].imshow(np.log1p(fft_magnitude), cmap='hot')
            axes[1, idx].set_title(f'{attack_name.upper()} Frequency Spectrum', fontsize=12)
            axes[1, idx].axis('off')
            plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def save_adv_images(self, results: Dict[str, Dict], output_dir: Path):
        """保存所有对抗样本图像"""
        output_dir.mkdir(parents=True, exist_ok=True)
        for attack_name, result in results.items():
            x_adv = self.denormalize(result["x_adv"])
            save_path = output_dir / f"adv_{attack_name}.png"
            save_image(x_adv, save_path)
            print(f"  Saved: {save_path}")


# ==================== 主执行流程 ====================
def load_model(device: torch.device, model_name: str = "resnet50") -> torch.nn.Module:
    """加载预训练模型"""
    print(f"Loading {model_name}...")
    try:
        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == "vgg19":
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except:
        # 回退到旧版API
        model = getattr(models, model_name)(pretrained=True)

    return model.eval().to(device)


def get_imagenet_transform() -> transforms.Compose:
    """ImageNet标准化预处理"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def get_mean_std(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """ImageNet均值方差"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="对抗攻击可视化分析器")

    # 输入设置
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg19"],
                        help="要攻击的模型")

    # 攻击参数
    parser.add_argument("--eps", type=float, default=8 / 255, help="扰动上限 (default: 8/255)")
    parser.add_argument("--alpha", type=float, default=2 / 255, help="PGD步长")
    parser.add_argument("--steps", type=int, default=20, help="PGD迭代步数")
    parser.add_argument("--cw_steps", type=int, default=100, help="CW攻击步数")
    parser.add_argument("--cw_c", type=float, default=1.0, help="CW攻击c参数")

    # 输出设置
    parser.add_argument("--output_dir", type=str, default="./attack_visualization",
                        help="输出目录")
    parser.add_argument("--save_images", action="store_true", help="保存对抗样本图片")

    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型和图像
    model = load_model(device, args.model)
    transform = get_imagenet_transform()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    img = Image.open(image_path).convert("RGB")
    x_clean = transform(img).unsqueeze(0).to(device)

    # 获取均值方差
    mean, std = get_mean_std(device)

    # 初始化可视化器
    visualizer = AttackVisualizer(model, device, mean, std)

    # 获取真实标签（使用模型预测作为伪标签）
    clean_pred_id, clean_pred_name, clean_prob = visualizer.predict(x_clean)
    print(f"\n{'=' * 60}")
    print(f"🖼️  输入图像: {image_path.name}")
    print(f"🎯 真实标签: {clean_pred_name} (class {clean_pred_id})")
    print(f"📊 置信度: {clean_prob:.4f}")
    print(f"{'=' * 60}")

    # 执行三种攻击
    print("\n⚔️  正在执行攻击...")
    results = {}

    # 1. FGSM攻击
    print("  执行 FGSM...")
    x_fgsm = fgsm_attack(model, x_clean, torch.tensor([clean_pred_id], device=device),
                         eps=args.eps, mean=mean, std=std)
    fgsm_pred = visualizer.predict(x_fgsm)
    results["fgsm"] = {
        "x_adv": x_fgsm,
        "pred_clean": (clean_pred_id, clean_pred_name, clean_prob),
        "pred_adv": fgsm_pred,
    }

    # 2. PGD攻击
    print("  执行 PGD...")
    x_pgd = pgd_linf_attack(model, x_clean, torch.tensor([clean_pred_id], device=device),
                            eps=args.eps, alpha=args.alpha, steps=args.steps,
                            mean=mean, std=std)
    pgd_pred = visualizer.predict(x_pgd)
    results["pgd"] = {
        "x_adv": x_pgd,
        "pred_clean": (clean_pred_id, clean_pred_name, clean_prob),
        "pred_adv": pgd_pred,
    }

    # 3. CW-L2攻击
    print("  执行 CW-L2...")
    cw_result = cw_l2_attack(model, x_clean, torch.tensor([clean_pred_id], device=device),
                             mean=mean, std=std, c=args.cw_c, kappa=0.0,
                             steps=args.cw_steps, lr=0.01)
    x_cw = cw_result.x_adv
    cw_pred = visualizer.predict(x_cw)
    results["cw"] = {
        "x_adv": x_cw,
        "pred_clean": (clean_pred_id, clean_pred_name, clean_prob),
        "pred_adv": cw_pred,
        "success": cw_result.success.item(),
    }

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 生成可视化 ====================
    print("\n📊 正在生成可视化...")

    # 1. 攻击效果网格图
    print("  生成攻击效果对比图...")
    visualizer.visualize_attack_grid(
        x_clean, results,
        save_path=output_dir / "attack_comparison.png"
    )

    # 2. 攻击轨迹分析（仅PGD）
    print("  生成攻击轨迹图...")
    visualizer.visualize_attack_trajectory(
        x_clean, clean_pred_id, "pgd",
        eps=args.eps, alpha=args.alpha, steps=args.steps,
        save_path=output_dir / "attack_trajectory.png"
    )

    # 3. 扰动分析
    print("  生成扰动分析图...")
    visualizer.visualize_perturbation_analysis(
        x_clean, results,
        save_path=output_dir / "perturbation_analysis.png"
    )

    # 4. 定量指标报告
    print("\n📈 定量评估指标:")
    print("-" * 80)
    for attack_name, result in results.items():
        print(f"\n{attack_name.upper()}攻击:")
        print(f"  预测变化: {result['pred_clean'][1]} ({result['pred_clean'][2]:.4f}) → "
              f"{result['pred_adv'][1]} ({result['pred_adv'][2]:.4f})")

        metrics = visualizer.calculate_metrics(x_clean, result["x_adv"])
        for metric, value in metrics.items():
            if "SSIM" in metric or "PSNR" in metric:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.6f}")

    # 5. 保存对抗样本图像
    if args.save_images:
        print("\n💾 保存对抗样本...")
        visualizer.save_adv_images(results, output_dir / "adversarial_images")

    # 6. 生成JSON报告
    report = {
        "image": str(image_path),
        "model": args.model,
        "clean_prediction": {
            "class_id": int(clean_pred_id),
            "class_name": clean_pred_name,
            "confidence": float(clean_prob)
        },
        "attacks": {
            name: {
                "predicted_class": int(result["pred_adv"][0]),
                "predicted_name": result["pred_adv"][1],
                "confidence": float(result["pred_adv"][2]),
                "success": result["pred_clean"][0] != result["pred_adv"][0],
                "metrics": visualizer.calculate_metrics(x_clean, result["x_adv"]),
            }
            for name, result in results.items()
        }
    }

    with open(output_dir / "attack_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ 所有结果已保存到: {output_dir}")
    print(f"📄 JSON报告: {output_dir / 'attack_report.json'}")


if __name__ == "__main__":
    main()