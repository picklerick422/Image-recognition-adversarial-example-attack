#!/usr/bin/env python3
"""
对抗样本攻击与防御实验框架 v2.1（最终修复版）
- 修复特征检测器阈值异常（返回合理范围0-10）
- 修复样本可视化维度错误
- 优化特征统计量计算
- 增加数值稳定性检查
"""

from __future__ import annotations

import argparse
import io
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

# ============= 屏蔽Unicode字体警告 =============
warnings.filterwarnings("ignore", "Glyph.*missing from font", UserWarning)


# ============= 攻击函数实现（保持不变） =============
def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """对批次数据进行归一化"""
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
    """FGSM单步攻击"""
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
    """PGD多步迭代攻击"""
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
    """反双曲正切函数，用于CW攻击"""
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
    """CW-L2优化攻击"""
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


# ============= 防御函数实现（保持不变） =============

# 输入预处理参数
_DEFENSE_USE_JPEG = False
_DEFENSE_JPEG_QUALITY = 75


def _defense_smoothing(x: torch.Tensor) -> torch.Tensor:
    """空间平滑：3×3平均池化"""
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def _defense_quantization(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    """颜色量化：将像素值离散化到有限级别"""
    x_clamped = x.clamp(0.0, 1.0)
    return torch.round(x_clamped * (levels - 1)) / (levels - 1)


def _jpeg_compress_batch(x: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG压缩：逐张压缩图像"""
    x_in = x.clamp(0.0, 1.0)
    device = x_in.device
    dtype = x_in.dtype
    x_cpu = x_in.detach().to("cpu")

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    out = torch.empty_like(x_cpu)

    for i in range(x_cpu.shape[0]):
        img = to_pil(x_cpu[i])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        img_jpeg = Image.open(buf).convert("RGB")
        out[i] = to_tensor(img_jpeg)

    return out.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def defend_input(x: torch.Tensor) -> torch.Tensor:
    """组合预处理防御：平滑 + 量化 + JPEG"""
    x01 = x.clamp(0.0, 1.0)
    x01 = _defense_smoothing(x01)
    x01 = _defense_quantization(x01, levels=16)
    if _DEFENSE_USE_JPEG:
        x01 = _jpeg_compress_batch(x01, quality=_DEFENSE_JPEG_QUALITY)
    return x01.clamp(0.0, 1.0)


# ========== 修复版特征提取函数（核心修复） ==========
def _layer3_features(model: torch.nn.Module, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    彻底修复版：提取ResNet50 layer3特征并计算合理的标量分数
    返回形状: [batch_size] 的标量分数（范围约0-10）
    """
    x_norm = normalize_batch(x, mean, std)

    with torch.no_grad():
        # 检查是否为标准ResNet结构
        if hasattr(model, "conv1") and hasattr(model, "layer3"):
            # 前向传播到layer3
            x1 = model.conv1(x_norm)
            x1 = model.bn1(x1)
            x1 = model.relu(x1)
            x1 = model.maxpool(x1)
            x1 = model.layer1(x1)
            x1 = model.layer2(x1)
            x1 = model.layer3(x1)

            # 计算每个样本的统计量（返回标量）
            batch_size = x1.shape[0]

            # 1. 特征图的空间平均L2范数（每个样本）
            # 形状: [batch_size, channels, h, w] -> [batch_size]
            feat_l2 = x1.pow(2).sum(dim=[1, 2, 3]).sqrt() / x1.shape[1]  # 除以通道数归一化

            # 2. 空间方差的平均值（每个样本）
            # 先计算每个通道在空间上的方差，再平均
            feat_var = x1.var(dim=[2, 3]).mean(dim=1)  # [batch_size]

            # 组合分数（范围通常在0-10之间）
            score = feat_l2 + feat_var * 0.1  # 降低方差权重

            # 防止数值过大
            score = torch.clamp(score, 0, 100)

            return score

        # 其他模型结构的fallback
        feats = model(x_norm)
        batch_size = feats.shape[0]

        if feats.ndim == 4:  # 特征图
            feat_l2 = feats.pow(2).sum(dim=[1, 2, 3]).sqrt() / feats.shape[1]
            feat_var = feats.var(dim=[2, 3]).mean(dim=1)
            return torch.clamp(feat_l2 + feat_var * 0.1, 0, 100)
        elif feats.ndim == 2:  # 全连接层输出
            return torch.clamp(feats.norm(dim=1), 0, 100)
        else:
            return torch.clamp(feats.view(batch_size, -1).norm(dim=1), 0, 100)


def is_adversarial_by_feature(
        model: torch.nn.Module,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        threshold: float,
) -> torch.Tensor:
    """
    修复版：基于layer3特征的对抗样本检测器
    返回形状: [batch_size] 的布尔值
    """
    scores = _layer3_features(model, x, mean, std)
    return scores > float(threshold)


def calibrate_feature_threshold(
        model: torch.nn.Module,
        image_paths: list[Path],
        *,
        device: torch.device,
        transform: transforms.Compose,
        mean: torch.Tensor,
        std: torch.Tensor,
        n: int = 100,
        quantile: float = 0.95,  # 使用0.95而非0.99
) -> float:
    """
    修复版：校准检测器阈值
    返回更合理的阈值（通常在0-10范围内）
    """
    scores = []
    num = min(int(n), len(image_paths))
    if num <= 0:
        raise ValueError("no calibration images available")

    print(f"正在校准检测器阈值（使用{num}张图像）...")
    for i, p in enumerate(image_paths[:num]):
        try:
            x = load_image(p, device, transform)
            # _layer3_features返回标量分数
            score = _layer3_features(model, x, mean, std)
            scores.append(score.detach().cpu())

            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{num} 张...")
        except Exception as e:
            print(f"⚠️  校准图像失败 {p.name}: {e}")
            continue

    if not scores:
        print("⚠️  没有成功处理任何校准图像，使用默认阈值")
        return 5.0

    # 合并所有分数
    all_scores = torch.cat(scores, dim=0)  # 形状: [num_samples]

    # 计算统计信息
    q = float(quantile)
    threshold = float(torch.quantile(all_scores, q).item())

    print(f"校准统计:")
    print(f"  分数范围: {all_scores.min():.4f} ~ {all_scores.max():.4f}")
    print(f"  平均值: {all_scores.mean():.4f}")
    print(f"  中位数: {torch.median(all_scores):.4f}")
    print(f"  {q * 100:.0f}%分位数（阈值）: {threshold:.4f}")

    # 验证阈值合理性
    if threshold > 50:
        print(f"⚠️  警告: 阈值过高({threshold:.4f})，可能检测到太多正常样本")
        print(f"    自动调整为: {threshold * 0.5:.4f}")
        return threshold * 0.5

    return max(threshold, 1.0)  # 确保阈值至少为1.0


# ============= 增强版可视化函数（保持之前的改进） =============

def plot_defense_heatmaps(results: dict, output_dir: Path, save_prefix: str = "defense_results"):
    """
    增强版防御实验可视化（修复排序问题，增加趋势分析）
    """
    try:
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("⚠️  缺少依赖: seaborn, pandas")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据整理（修复EPS排序问题）
    data = []
    for (attack_name, eps), stats in results.items():
        count = max(1, stats['count'])
        data.append({
            'Attack': attack_name.upper(),
            'Eps': float(eps),
            'Attack_Success': stats['attack_success'] / count,
            'Preproc_Defense_Acc': stats['defense_preproc_success'] / count,
            'Detector_Clean_Pass': 1.0 - stats['detector_flags_clean'] / count,
            'Detector_Adv_Flag': stats['detector_flags_adv'] / count,
            'Bypass_Detection': stats['detector_attack_success'] / count,
        })

    df = pd.DataFrame(data)
    df = df.sort_values(['Attack', 'Eps'])

    # 图表1：攻击成功率趋势图
    plt.figure(figsize=(12, 6))
    for attack in df['Attack'].unique():
        subset = df[df['Attack'] == attack]
        plt.plot(subset['Eps'], subset['Attack_Success'], 'o-',
                 label=attack, linewidth=2.5, markersize=8)

    plt.xlabel('Perturbation Budget (eps)', fontsize=12, fontweight='bold')
    plt.ylabel('Attack Success Rate', fontsize=12, fontweight='bold')
    plt.title('Attack Success Rate vs. Perturbation Strength', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_attack_trend.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 图表2：防御性能矩阵
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Defense Performance Matrix', fontsize=16, fontweight='bold')

    metrics = ['Preproc_Defense_Acc', 'Detector_Adv_Flag', 'Bypass_Detection']
    titles = ['Preprocessing Defense\nAccuracy', 'Detector Flag\nRate', 'Bypass Detection\nSuccess Rate']
    cmaps = ['RdYlGn', 'YlOrRd', 'YlOrRd']

    for idx, (metric, title, cmap) in enumerate(zip(metrics, titles, cmaps)):
        ax = axes[idx // 2, idx % 2]
        pivot = df.pivot(index='Eps', columns='Attack', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                    cbar_kws={'label': 'Rate'}, linewidths=.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Attack Method')
        ax.set_ylabel('Perturbation (eps)')

    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_defense_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ 可视化已保存到: {output_dir}")


def visualize_attack_samples(
        model: torch.nn.Module,
        image_paths: list[Path],
        output_dir: Path,
        n_samples: int = 5,
        eps: float = 8 / 255
):
    """
    修复版：可视化原始、对抗、防御后的样本对比
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  缺少依赖: matplotlib")
        return

    device = next(model.parameters()).device
    transform = get_transform()
    mean, std = get_mean_std(device, torch.float32)

    n_samples = min(n_samples, len(image_paths))
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    # 处理n_samples=1的情况
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Attack & Defense Samples (ε={eps:.3f})', fontsize=16, fontweight='bold')

    for idx, path in enumerate(image_paths[:n_samples]):
        try:
            # 加载图像
            x = load_image(path, device, transform)

            # 检查预测是否正确
            logits_clean = model(normalize_batch(x, mean, std))
            y_true = logits_clean.argmax(dim=1)
            pred_clean = y_true.item()

            # 只处理预测正确的样本
            if logits_clean[0, pred_clean].item() < 0.5:
                print(f"⚠️  跳过置信度过低的样本: {path.name}")
                continue

            # 生成对抗样本
            if eps > 0:
                x_adv = pgd_linf_attack(model, x, y_true, eps=eps, alpha=eps / 4, steps=10, mean=mean, std=std)
            else:
                x_adv = x.clone()

            # 应用防御
            x_def = defend_input(x_adv)

            # 获取预测结果
            with torch.no_grad():
                pred_adv = model(normalize_batch(x_adv, mean, std)).argmax(dim=1).item()
                pred_def = model(normalize_batch(x_def, mean, std)).argmax(dim=1).item()

            # 反归一化用于显示
            def denormalize(img):
                return img * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

            # 确保移除batch维度并转换为CPU
            images = [
                x[0] if x.ndim == 4 else x,  # 确保3D
                x_adv[0] if x_adv.ndim == 4 else x_adv,
                x_def[0] if x_def.ndim == 4 else x_def,
                torch.abs(x_adv - x).sum(dim=1, keepdim=True)[0]  # 先sum再取batch
            ]

            titles = [
                f'Clean\nPred: {pred_clean}\nConf: {logits_clean[0, pred_clean].item():.3f}',
                f'Adversarial\nPred: {pred_adv}',
                f'Defended\nPred: {pred_def}',
                f'Perturbation\nMagnitude'
            ]

            for col in range(4):
                ax = axes[idx, col]
                img = images[col]

                if col < 3:
                    img = denormalize(img)
                    # 确保是3D张量并转换为numpy
                    if img.ndim == 4:  # 如果还是4D，强制取第一个样本
                        img = img[0]
                    img_np = img.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                    ax.imshow(img_np)
                else:
                    # 扰动图（单通道）
                    perturb = img.cpu().numpy()
                    im = ax.imshow(perturb[0], cmap='hot')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_title(titles[col], fontsize=10)
                ax.axis('off')

        except Exception as e:
            print(f"⚠️  样本可视化失败 {path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    plt.tight_layout()
    plt.savefig(output_dir / "attack_samples.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ 样本可视化已保存: {output_dir / 'attack_samples.png'}")


# ============= 评估函数（保持不变） =============
def run_attack(
        attack_name: str,
        model: torch.nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float,
        alpha: float,
        steps: int,
        cw_c: float,
        cw_kappa: float,
        cw_steps: int,
        cw_lr: float,
) -> torch.Tensor:
    """运行指定攻击"""
    if attack_name == "fgsm":
        return fgsm_attack(model, x, y_true, eps=eps, mean=mean, std=std)
    if attack_name == "pgd":
        return pgd_linf_attack(
            model,
            x,
            y_true,
            eps=eps,
            alpha=alpha,
            steps=steps,
            mean=mean,
            std=std,
        )
    cw_res = cw_l2_attack(
        model,
        x,
        y_true,
        mean=mean,
        std=std,
        c=cw_c,
        kappa=cw_kappa,
        steps=cw_steps,
        lr=cw_lr,
        targeted=False,
        y_target=None,
    )
    return cw_res.x_adv


def evaluate_defenses(
        model: torch.nn.Module,
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
        detector_threshold: float,
) -> dict:
    """评估防御效果"""
    # 干净样本预测
    logits_clean = model(normalize_batch(x, mean, std))
    pred_clean = logits_clean.argmax(dim=1)
    clean_correct = (pred_clean == y_true).item()

    # 生成对抗样本
    x_adv = run_attack(
        attack_name,
        model,
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

    # 对抗样本预测
    logits_adv = model(normalize_batch(x_adv, mean, std))
    pred_adv = logits_adv.argmax(dim=1)
    attack_success = int(pred_adv != y_true)

    # 预处理防御
    x_adv_def = defend_input(x_adv)
    logits_def = model(normalize_batch(x_adv_def, mean, std))
    pred_def = logits_def.argmax(dim=1)
    defense_preproc_success = int(pred_def == y_true)

    # 检测器评估
    detected_adv = is_adversarial_by_feature(model, x_adv, mean, std, detector_threshold)
    detector_flags_adv = int(detected_adv.item())

    detected_clean = is_adversarial_by_feature(model, x, mean, std, detector_threshold)
    detector_flags_clean = int(detected_clean.item())

    # 绕检测器攻击成功率
    detector_attack_success = int((attack_success == 1) and (detector_flags_adv == 0))

    return {
        "clean_correct": clean_correct,
        "attack_success": attack_success,
        "defense_preproc_success": defense_preproc_success,
        "detector_flags_clean": detector_flags_clean,
        "detector_flags_adv": detector_flags_adv,
        "detector_attack_success": detector_attack_success,
    }


def load_image(path: Path, device: torch.device, transform: transforms.Compose) -> torch.Tensor:
    """加载并预处理图像"""
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)
    return x.to(device)


def get_transform() -> transforms.Compose:
    """获取ImageNet预处理"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def get_mean_std(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """获取ImageNet归一化参数"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
    return mean, std


def load_model(device: torch.device, model_type: str = "standard") -> torch.nn.Module:
    """加载模型（标准或鲁棒）"""
    if model_type == "robust":
        try:
            from robustbench import load_model as rb_load_model
            model = rb_load_model(
                model_name="Engstrom2019Robustness",
                dataset="imagenet",
                threat_model="Linf",
            ).eval()
        except Exception as e:
            raise RuntimeError(
                "Failed to load robust model from RobustBench. "
                "Ensure robustbench and autoattack are installed."
            ) from e
    else:
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        except:
            model = models.resnet50(pretrained=True).eval()
    return model.to(device)


# ============= 主实验流程（增强版） =============
def main() -> None:
    """主函数：运行攻击与防御实验"""
    parser = argparse.ArgumentParser(
        description="对抗样本攻击与防御实验框架 v2.1（修复版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python defense_experiments.py --image_dir ./imagenet_val/ --model_type standard\n"
               "  python defense_experiments.py --image ./picture/example.jpg --eps 0.062745 --use_jpeg\n"
    )

    # 模型与数据输入
    parser.add_argument("--model_type", type=str, choices=["standard", "robust"], default="standard")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image", type=str, default="example.jpg")

    # 攻击配置
    parser.add_argument("--attacks", type=str, nargs="+", default=["fgsm", "pgd", "cw"],
                        choices=["fgsm", "pgd", "cw"])
    parser.add_argument("--eps_list", type=float, nargs="+", default=[4 / 255, 8 / 255, 16 / 255])
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cw_c", type=float, default=1.0)
    parser.add_argument("--cw_kappa", type=float, default=0.0)
    parser.add_argument("--cw_steps", type=int, default=100)
    parser.add_argument("--cw_lr", type=float, default=0.01)

    # 检测器配置
    parser.add_argument("--detector_threshold", type=float, default=None)
    parser.add_argument("--calibrate_dir", type=str, default=None)
    parser.add_argument("--calibrate_n", type=int, default=100)
    parser.add_argument("--calibrate_quantile", type=float, default=0.95)

    # 预处理防御配置
    parser.add_argument("--use_jpeg", action="store_true")
    parser.add_argument("--jpeg_quality", type=int, default=75)

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./defense_results")
    parser.add_argument("--viz_samples", type=int, default=5,
                        help="要可视化的攻击样本数量（0表示禁用）")

    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(device, model_type=args.model_type)
    transform = get_transform()

    # 获取图像列表
    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise SystemExit(f"image_dir not found: {image_dir}")
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not image_paths:
            raise SystemExit(f"no images found in {image_dir}")
        print(f"加载图像目录: {image_dir} ({len(image_paths)}张图像)")
    else:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise SystemExit(f"image not found: {image_path}")
        image_paths = [image_path]
        print(f"加载单张图像: {image_path}")

    mean, std = get_mean_std(device, torch.float32)
    if args.model_type == "robust":
        mean = torch.zeros_like(mean)
        std = torch.ones_like(std)

    # 配置预处理防御
    global _DEFENSE_USE_JPEG, _DEFENSE_JPEG_QUALITY
    _DEFENSE_USE_JPEG = bool(args.use_jpeg)
    _DEFENSE_JPEG_QUALITY = int(args.jpeg_quality)

    # 校准检测器阈值
    if args.calibrate_dir is not None:
        calib_dir = Path(args.calibrate_dir)
        if not calib_dir.is_dir():
            raise SystemExit(f"calibrate_dir not found: {calib_dir}")
        calib_paths = sorted([p for p in calib_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not calib_paths:
            raise SystemExit(f"no images found in calibrate_dir: {calib_dir}")
        detector_threshold = calibrate_feature_threshold(
            model, calib_paths, device=device, transform=transform, mean=mean, std=std,
            n=int(args.calibrate_n), quantile=float(args.calibrate_quantile)
        )
        print(f"使用校准阈值: {detector_threshold:.4f}")
    else:
        if args.detector_threshold is not None:
            detector_threshold = float(args.detector_threshold)
            print(f"使用指定阈值: {detector_threshold:.4f}")
        else:
            detector_threshold = calibrate_feature_threshold(
                model, image_paths, device=device, transform=transform, mean=mean, std=std,
                n=min(100, len(image_paths)), quantile=float(args.calibrate_quantile)
            )
            print(f"自动校准阈值: {detector_threshold:.4f}")

    # 运行实验
    results = {}
    print("\n" + "=" * 60)
    print("开始攻击与防御实验...")
    print("=" * 60)

    for attack_name in args.attacks:
        if attack_name not in {"fgsm", "pgd", "cw"}:
            continue
        for eps in args.eps_list:
            key = (attack_name, float(eps))
            stats = {
                "clean_correct": 0, "attack_success": 0, "defense_preproc_success": 0,
                "detector_flags_clean": 0, "detector_flags_adv": 0, "detector_attack_success": 0, "count": 0,
            }

            print(f"\n【{attack_name.upper()} Attack | eps={eps:.5f}】")

            for path in image_paths:
                x = load_image(path, device, transform)
                logits_clean = model(normalize_batch(x, mean, std))
                y_true = logits_clean.argmax(dim=1)

                eval_stats = evaluate_defenses(
                    model=model, x=x, y_true=y_true, mean=mean, std=std,
                    attack_name=attack_name, eps=float(eps), alpha=float(args.alpha),
                    steps=int(args.steps), cw_c=float(args.cw_c), cw_kappa=float(args.cw_kappa),
                    cw_steps=int(args.cw_steps), cw_lr=float(args.cw_lr),
                    detector_threshold=float(detector_threshold),
                )

                for k in stats:
                    if k == "count": continue
                    stats[k] += int(eval_stats[k])
                stats["count"] += 1

            results[key] = stats

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)

    for (attack_name, eps), stats in sorted(results.items()):
        count = max(1, stats["count"])
        print(
            f"attack={attack_name}, eps={eps:.5f}, "
            f"attack_success={stats['attack_success'] / count:.3f}, "
            f"preproc_defense_acc={stats['defense_preproc_success'] / count:.3f}, "
            f"detector_clean_pass_rate={1.0 - stats['detector_flags_clean'] / count:.3f}, "
            f"detector_adv_flag_rate={stats['detector_flags_adv'] / count:.3f}, "
            f"detector_attack_success={stats['detector_attack_success'] / count:.3f}"
        )

    # 生成可视化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("📊 生成攻击样本可视化...")
    print("=" * 60)

    if args.viz_samples > 0:
        visualize_attack_samples(
            model=model, image_paths=image_paths, output_dir=output_dir,
            n_samples=int(args.viz_samples), eps=float(args.eps_list[1]) if len(args.eps_list) > 1 else 8 / 255
        )

    print("\n" + "=" * 60)
    print("📊 生成防御效果热力图...")
    print("=" * 60)

    plot_defense_heatmaps(results, output_dir, save_prefix="defense_results")

    print("\n🎉 所有实验完成！结果已保存到:", output_dir)


# ============= 入口 =============
if __name__ == "__main__":
    # 检查依赖
    try:
        import seaborn
        import pandas
    except ImportError:
        print("❌ 错误: 必须安装 seaborn 和 pandas")
        print("    pip install seaborn pandas")
        exit(1)

    print("=" * 60)
    print("对抗样本攻击与防御实验框架 v2.1（最终修复版）")
    print("=" * 60)

    main()