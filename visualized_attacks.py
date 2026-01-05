#!/usr/bin/env python3
"""
对抗攻击可视化分析器 v6.0 (终极修复版)
- ✅ 完全复刻 ResNet.py 的模型加载方式 (pretrained=True)
- ✅ 使用旧版 IMAGENET1K_V1 权重，与 ResNet.py 保持一致
- ✅ 全局定义CPU张量 mean/std，每次动态转换
- ✅ 消除所有与 ResNet.py 的实现差异
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

# ============= 屏蔽Unicode字体警告 =============
warnings.filterwarnings("ignore", "Glyph.*missing from font", UserWarning)


# ============= 攻击函数实现（保持不变） =============
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
            success = (logits.argmax(dim=1) != y_true)

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


# ============= 全局变量（与ResNet.py完全一致） =============
# ✅ 在模块级别定义CPU张量，与ResNet.py完全相同
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ============= 可视化核心类 =============
class AttackVisualizer:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 imagenet_classes: list = None):
        self.model = model
        self.device = device

        # ✅ 使用 torchmetrics 计算SSIM
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self.ssim_available = True
        except ImportError:
            print("⚠️  warning: torchmetrics未安装，SSIM将无法计算")
            print("  请运行: pip install torchmetrics")
            self.ssim_available = False

        # 加载类别标签
        self.imagenet_classes = imagenet_classes or self._load_imagenet_classes()

    def _load_imagenet_classes(self) -> list:
        """加载ImageNet类别标签"""
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            classes = urllib.request.urlopen(url, timeout=5).read().decode().strip().split('\n')
            return classes
        except:
            return [f"class_{i}" for i in range(1000)]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化到[0,1]"""
        # ✅ 使用全局CPU张量，每次动态转换
        mean_device = mean.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std_device = std.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return torch.clamp(x * std_device + mean_device, 0.0, 1.0)

    def predict(self, x: torch.Tensor) -> Tuple[int, str, float]:
        """✅ 与ResNet.py完全一致的推理逻辑"""
        with torch.no_grad():
            # 使用全局CPU张量，每次动态转换（与ResNet.py完全相同）
            mean_device = mean.to(device=x.device, dtype=x.dtype)
            std_device = std.to(device=x.device, dtype=x.dtype)

            logits = self.model(normalize_batch(x, mean_device, std_device))
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

        # ✅ SSIM计算
        if self.ssim_available:
            ssim_score = self.ssim_metric(x_clean, x_adv).item()
        else:
            ssim_score = 0.0

        # PSNR计算
        mse = torch.mean(diff ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100.0

        # 扰动像素比例
        perturbed_pixels = (torch.abs(diff) > 1 / 255).float().mean().item() * 100

        # 高频扰动比例
        fft_diff = torch.fft.fft2(diff[0])
        high_freq_ratio = (torch.abs(fft_diff) > torch.mean(torch.abs(fft_diff))).float().mean().item() * 100

        return {
            "L∞ (pixel)": l_inf,
            "L2": l_2,
            "L1": l_1,
            "SSIM": ssim_score,
            "PSNR": psnr,
            "Perturbed Pixels %": perturbed_pixels,
            "High Freq Ratio %": high_freq_ratio,
        }

    def visualize_attack_grid(self, x_clean: torch.Tensor,
                              results: Dict[str, Dict],
                              save_path: Path = None):
        """生成攻击效果图网格"""
        n_attacks = len(results)
        fig = plt.figure(figsize=(4 * 3, 4 * n_attacks))

        x_clean_denorm = self.denormalize(x_clean).squeeze()

        for idx, (attack_name, result) in enumerate(results.items()):
            x_adv = result["x_adv"]
            x_adv_denorm = self.denormalize(x_adv).squeeze()
            diff = (x_adv - x_clean).squeeze()

            clean_id = result["pred_clean"][0]
            adv_id = result["pred_adv"][0]
            success = "SUCCESS" if clean_id != adv_id else "FAILED"

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

            fig.text(0.5, 1 - (idx * (1 / n_attacks) - 0.02),
                     f"{attack_name.upper()} Attack - {success}",
                     ha='center', va='top', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  已保存: {save_path}")

    def _plot_image(self, ax, img_tensor, title):
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    def _plot_image_pair(self, ax, img1, img2, title1, title2):
        img_np = torch.cat([img1, img2], dim=2).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(f"{title1} vs {title2}", fontsize=11)
        ax.axis('off')
        h, w = img1.shape[1:]
        ax.axvline(x=w, color='white', linewidth=2)

    def visualize_attack_trajectory(self, x_clean: torch.Tensor, y_true: int,
                                    attack_name: str, eps: float, alpha: float, steps: int,
                                    save_path: Path = None):
        """动态展示攻击过程"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        traj_probs = []
        traj_perturbs = []

        x_orig = x_clean.detach()
        x_adv = x_orig.clone()
        if attack_name == "pgd":
            x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0, 1)

        for step in range(steps + 1):
            with torch.no_grad():
                logits = self.model(normalize_batch(x_adv, mean, std))
                probs = F.softmax(logits, dim=1)
                traj_probs.append(probs[0, [y_true, 805]].cpu().numpy())
                traj_perturbs.append(torch.norm(x_adv - x_orig).item())

            if step > 0 and attack_name == "pgd":
                x_adv = x_adv.detach().clone().requires_grad_(True)
                logits = self.model(normalize_batch(x_adv, mean, std))
                loss = F.cross_entropy(logits, torch.tensor([y_true], device=self.device))
                grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
                x_adv = x_adv + alpha * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0, 1)
            elif attack_name == "fgsm" and step == 1:
                x_in = x_orig.clone().requires_grad_(True)
                logits = self.model(normalize_batch(x_in, mean, std))
                loss = F.cross_entropy(logits, torch.tensor([y_true], device=self.device))
                grad = torch.autograd.grad(loss, x_in, only_inputs=True)[0]
                x_adv = x_orig + eps * grad.sign()
                x_adv = x_adv.clamp(0, 1)
                break

        traj_probs = np.array(traj_probs)

        ax1.plot(traj_probs[:, 0], label='Original Class', color='green', linewidth=2, marker='o', markersize=3)
        ax1.plot(traj_probs[:, 1], label='Target Class', color='red', linewidth=2, marker='x', markersize=3)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')
        ax1.set_xlabel('Attack Step', fontsize=12)
        ax1.set_ylabel('Prediction Probability', fontsize=12)
        ax1.set_title(f'{attack_name.upper()} Attack Trajectory (Eps={eps:.5f})', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        ax2.plot(traj_perturbs, color='purple', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Attack Step', fontsize=12)
        ax2.set_ylabel('L2 Perturbation', fontsize=12)
        ax2.set_title('Perturbation Growth', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"  已保存: {save_path}")

    def visualize_perturbation_analysis(self, x_clean: torch.Tensor,
                                        results: Dict[str, Dict],
                                        save_path: Path = None):
        """扰动的频域和空域分析"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Perturbation Spatial & Frequency Analysis', fontsize=16, fontweight='bold')

        for idx, (attack_name, result) in enumerate(results.items()):
            diff = (result["x_adv"] - x_clean).squeeze().cpu().numpy()
            axes[0, idx].hist(diff.flatten(), bins=50, alpha=0.7,
                              color=['red', 'blue', 'purple'][idx],
                              range=(-0.1, 0.1))
            axes[0, idx].set_title(f'{attack_name.upper()} Distribution', fontsize=12)
            axes[0, idx].set_xlabel('Perturbation Value')
            axes[0, idx].set_ylabel('Frequency')
            axes[0, idx].grid(True, alpha=0.3)

        for idx, (attack_name, result) in enumerate(results.items()):
            diff = (result["x_adv"] - x_clean).squeeze().cpu().numpy()
            fft_diff = np.fft.fft2(diff.transpose(1, 2, 0).mean(axis=2))
            fft_magnitude = np.abs(np.fft.fftshift(fft_diff))

            im = axes[1, idx].imshow(np.log1p(fft_magnitude), cmap='hot')
            axes[1, idx].set_title(f'{attack_name.upper()} Frequency', fontsize=12)
            axes[1, idx].axis('off')
            plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"  已保存: {save_path}")

    def save_adv_images(self, results: Dict[str, Dict], output_dir: Path):
        """保存所有对抗样本图像"""
        output_dir.mkdir(parents=True, exist_ok=True)
        for attack_name, result in results.items():
            x_adv = self.denormalize(result["x_adv"])
            save_path = output_dir / f"adv_{attack_name}.png"
            save_image(x_adv, save_path)
            print(f"    {save_path}")


# ============= 模型和数据加载 =============
def load_model(device: torch.device, model_name: str = "resnet50") -> torch.nn.Module:
    """✅ 加载预训练模型（与ResNet.py完全一致）"""
    print(f"Loading {model_name}...")

    # ✅ 核心修复：使用pretrained=True而非weights=...，确保加载相同权重版本
    model = getattr(models, model_name)(pretrained=True).eval()

    return model.to(device)


def get_imagenet_transform() -> transforms.Compose:
    """ImageNet标准化预处理"""
    return transform  # 使用全局变量


def get_device():
    """获取计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============= 主执行流程 =============
def main():
    parser = argparse.ArgumentParser(
        description="对抗攻击可视化分析器 v6.0 (最终权重修复版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python visualize_attacks.py --image picture/example.jpg\n"
               "  python visualize_attacks.py --image picture/example.jpg --eps 0.062745 --steps 40\n"
               "  python visualize_attacks.py --image picture/example.jpg --cw_c 0.1 --cw_steps 500"
    )

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
    device = get_device()
    print(f"使用设备: {device}")
    if not torch.cuda.is_available():
        print("⚠️  警告: 未检测到CUDA，CPU模式会非常慢")

    # 加载模型（✅ 使用与ResNet.py完全相同的方式）
    model = load_model(device, args.model)

    # 加载图像
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    img = Image.open(image_path).convert("RGB")
    x_clean = transform(img).unsqueeze(0).to(device)

    # 初始化可视化器（✅ 不传mean/std，使用全局变量）
    visualizer = AttackVisualizer(model, device)

    # 获取真实标签（现在与ResNet.py完全一致）
    clean_pred_id, clean_pred_name, clean_prob = visualizer.predict(x_clean)
    print(f"\n{'=' * 60}")
    print(f"🖼️  输入图像: {image_path.name}")
    print(f"🎯 真实标签: {clean_pred_name} (class {clean_pred_id})")
    print(f"📊 置信度: {clean_prob:.4f}")  # 必须显示0.997
    print(f"{'=' * 60}")

    # 执行三种攻击（✅ 传递全局CPU张量给攻击函数）
    print("\n⚔️  正在执行攻击...")
    results = {}

    for attack_name in ["fgsm", "pgd", "cw"]:
        print(f"  执行 {attack_name.upper()}...")

        if attack_name == "fgsm":
            x_adv = fgsm_attack(model, x_clean, torch.tensor([clean_pred_id], device=device),
                                eps=args.eps, mean=mean, std=std)  # ✅ 传递全局CPU张量
        elif attack_name == "pgd":
            x_adv = pgd_linf_attack(model, x_clean, torch.tensor([clean_pred_id], device=device),
                                    eps=args.eps, alpha=args.alpha, steps=args.steps,
                                    mean=mean, std=std)  # ✅ 传递全局CPU张量
        else:  # cw
            cw_result = cw_l2_attack(model, x_clean, torch.tensor([clean_pred_id], device=device),
                                     mean=mean, std=std, c=args.cw_c, kappa=0.0,
                                     steps=args.cw_steps, lr=0.01)  # ✅ 传递全局CPU张量
            x_adv = cw_result.x_adv

        # 获取预测结果（使用修复后的predict方法）
        adv_pred = visualizer.predict(x_adv)

        results[attack_name] = {
            "x_adv": x_adv,
            "pred_clean": (clean_pred_id, clean_pred_name, clean_prob),
            "pred_adv": adv_pred,
        }

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============= 生成可视化 =============
    print("\n📊 正在生成可视化...")

    # 1. 攻击效果对比图
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
        clean_info = result["pred_clean"]
        adv_info = result["pred_adv"]

        # 判断攻击是否成功
        success = "SUCCESS" if clean_info[0] != adv_info[0] else "FAILED"
        print(f"\n{attack_name.upper()}攻击 [{success}]:")
        print(f"  预测变化: {clean_info[1]} ({clean_info[2]:.4f}) → "
              f"{adv_info[1]} ({adv_info[2]:.4f})")

        metrics = visualizer.calculate_metrics(x_clean, result["x_adv"])
        for metric, value in metrics.items():
            if isinstance(value, float):
                if "SSIM" in metric or "PSNR" in metric:
                    print(f"  {metric:.<25} {value:.4f}")
                else:
                    print(f"  {metric:.<25} {value:.6f}")
            else:
                print(f"  {metric:.<25} {value}")

    # 5. 保存对抗样本图像
    if args.save_images:
        print("\n💾 保存对抗样本...")
        visualizer.save_adv_images(results, output_dir / "adversarial_images")

    # 6. 生成JSON报告
    report = {
        "image": str(image_path.absolute()),
        "model": args.model,
        "clean_prediction": {
            "class_id": int(clean_pred_id),
            "class_name": clean_pred_name,
            "confidence": float(clean_prob)
        },
        "params": {
            "eps": float(args.eps),
            "alpha": float(args.alpha),
            "steps": int(args.steps),
            "cw_c": float(args.cw_c),
            "cw_steps": int(args.cw_steps)
        },
        "attacks": {
            name: {
                "predicted_class": int(result["pred_adj"][0]),
                "predicted_name": result["pred_adv"][1],
                "confidence": float(result["pred_adv"][2]),
                "success": result["pred_clean"][0] != result["pred_adv"][0],
                "metrics": visualizer.calculate_metrics(x_clean, result["x_adv"]),
            }
            for name, result in results.items()
        }
    }

    # 保存报告时处理numpy数据类型
    def json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    with open(output_dir / "attack_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=json_serializable, ensure_ascii=False)

    print(f"\n✅ 所有结果已保存到: {output_dir}")
    print(f"📄 JSON报告: {output_dir / 'attack_report.json'}")
    print(f"\n🎉 完成！请检查输出目录中的PNG图片。")


# ============= 入口 & 依赖检查 =============
if __name__ == "__main__":
    # 检查关键依赖
    try:
        import torchmetrics
    except ImportError:
        print("❌ 错误: torchmetrics未安装")
        print("   请运行: pip install torchmetrics")
        exit(1)

    try:
        import matplotlib
    except ImportError:
        print("❌ 错误: matplotlib未安装")
        print("   请运行: pip install matplotlib")
        exit(1)

    main()