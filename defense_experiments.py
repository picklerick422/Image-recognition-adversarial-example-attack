import argparse
import io
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from attack import cw_l2_attack, fgsm_attack, normalize_batch, pgd_linf_attack


def load_model(device: torch.device, model_type: str = "standard") -> torch.nn.Module:
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
                "Ensure robustbench and autoattack are installed and model_name is valid."
            ) from e
    else:
        try:
            from torchvision.models import ResNet50_Weights

            model = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        except Exception:
            model = models.resnet50(pretrained=True).eval()
    return model.to(device)


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def load_image(path: Path, device: torch.device, transform: transforms.Compose) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)
    return x.to(device)


def get_mean_std(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
    return mean, std


def predict(model: torch.nn.Module, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(normalize_batch(x, mean, std))
    return logits


_DEFENSE_USE_JPEG = False
_DEFENSE_JPEG_QUALITY = 75


def _defense_smoothing(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def _defense_quantization(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    x_clamped = x.clamp(0.0, 1.0)
    return torch.round(x_clamped * (levels - 1)) / (levels - 1)


def _jpeg_compress_batch(x: torch.Tensor, quality: int) -> torch.Tensor:
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
    """输入预处理防御：组合平滑 + 量化 + JPEG"""
    x01 = x.clamp(0.0, 1.0)
    x01 = _defense_smoothing(x01)
    x01 = _defense_quantization(x01, levels=16)
    if _DEFENSE_USE_JPEG:
        x01 = _jpeg_compress_batch(x01, quality=_DEFENSE_JPEG_QUALITY)
    return x01.clamp(0.0, 1.0)


def _layer3_features(model: torch.nn.Module, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x_norm = normalize_batch(x, mean, std)
    with torch.no_grad():
        if hasattr(model, "conv1") and hasattr(model, "layer3"):
            x1 = model.conv1(x_norm)
            x1 = model.bn1(x1)
            x1 = model.relu(x1)
            x1 = model.maxpool(x1)
            x1 = model.layer1(x1)
            x1 = model.layer2(x1)
            x1 = model.layer3(x1)
            return x1
        feats = model(x_norm)
        if feats.ndim == 2:
            b, c = feats.shape
            return feats.view(b, c, 1, 1)
        if feats.ndim == 4:
            return feats
        b = feats.shape[0]
        return feats.view(b, -1, 1, 1)


def is_adversarial_by_feature(
    model: torch.nn.Module,
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """基于 ResNet50 layer3 特征范数的检测器"""
    feats = _layer3_features(model, x, mean, std)
    b = feats.shape[0]
    l2 = feats.view(b, -1).pow(2).sum(dim=1).sqrt()
    ch_var = feats.var(dim=(2, 3), unbiased=False).mean(dim=1)
    score = l2 + ch_var
    return score > float(threshold)


def calibrate_feature_threshold(
    model: torch.nn.Module,
    image_paths: list[Path],
    *,
    device: torch.device,
    transform: transforms.Compose,
    mean: torch.Tensor,
    std: torch.Tensor,
    n: int = 100,
    quantile: float = 0.99,
) -> float:
    scores = []
    num = min(int(n), len(image_paths))
    if num <= 0:
        raise ValueError("no calibration images available")

    for p in image_paths[:num]:
        x = load_image(p, device, transform)
        feats = _layer3_features(model, x, mean, std)
        l2 = feats.view(1, -1).pow(2).sum(dim=1).sqrt()
        ch_var = feats.var(dim=(2, 3), unbiased=False).mean(dim=1)
        score = (l2 + ch_var).detach().to("cpu")
        scores.append(score)

    s = torch.cat(scores, dim=0).to(dtype=torch.float32)
    q = float(quantile)
    q = max(0.0, min(1.0, q))
    return float(torch.quantile(s, q).item())


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
    logits_clean = predict(model, x, mean, std)
    pred_clean = logits_clean.argmax(dim=1)
    clean_correct = (pred_clean == y_true).item()

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

    logits_adv = predict(model, x_adv, mean, std)
    pred_adv = logits_adv.argmax(dim=1)
    attack_success = int(pred_adv != y_true)

    x_adv_def = defend_input(x_adv)
    logits_def = predict(model, x_adv_def, mean, std)
    pred_def = logits_def.argmax(dim=1)
    defense_preproc_success = int(pred_def == y_true)

    detected_adv = is_adversarial_by_feature(model, x_adv, mean, std, detector_threshold)
    detector_flags_adv = int(detected_adv.item())

    detected_clean = is_adversarial_by_feature(model, x, mean, std, detector_threshold)
    detector_flags_clean = int(detected_clean.item())

    detector_attack_success = int((attack_success == 1) and (detector_flags_adv == 0))

    return {
        "clean_correct": clean_correct,
        "attack_success": attack_success,
        "defense_preproc_success": defense_preproc_success,
        "detector_flags_clean": detector_flags_clean,
        "detector_flags_adv": detector_flags_adv,
        "detector_attack_success": detector_attack_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["standard", "robust"],
        default="standard",
    )  # 选择使用标准模型还是鲁棒模型
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
    )  # 包含待测试图片的文件夹路径（与 --image 二选一）
    parser.add_argument(
        "--image",
        type=str,
        default="example.jpg",
    )  # 单张待测试图片路径（当未指定 --image_dir 时生效）
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["fgsm", "pgd", "cw"],
    )  # 要运行的攻击方法列表
    parser.add_argument(
        "--eps_list",
        type=float,
        nargs="+",
        default=[4 / 255, 8 / 255, 16 / 255],
    )  # FGSM/PGD 等 \(L_\infty\) 攻击的扰动半径列表
    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
    )  # PGD 每一步的步长
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )  # 迭代型攻击（如 PGD）的迭代步数
    parser.add_argument(
        "--cw_c",
        type=float,
        default=1.0,
    )  # CW 攻击中损失函数前的权重系数 \(c\)
    parser.add_argument(
        "--cw_kappa",
        type=float,
        default=0.0,
    )  # CW 攻击中的置信度参数 \(\kappa\)
    parser.add_argument(
        "--cw_steps",
        type=int,
        default=100,
    )  # CW 攻击的最大迭代步数
    parser.add_argument(
        "--cw_lr",
        type=float,
        default=0.01,
    )  # CW 攻击中优化变量的学习率
    parser.add_argument(
        "--detector_threshold",
        type=float,
        default=None,
    )  # 特征空间检测器判定对抗样本的阈值（None 表示使用默认或自动校准）
    parser.add_argument(
        "--calibrate_dir",
        type=str,
        default=None,
    )  # 用于阈值校准的干净样本目录
    parser.add_argument(
        "--calibrate_n",
        type=int,
        default=100,
    )  # 用于校准的样本数量上限
    parser.add_argument(
        "--calibrate_quantile",
        type=float,
        default=0.99,
    )  # 设置阈值时使用的分位数（越大越保守）
    parser.add_argument(
        "--use_jpeg",
        action="store_true",
    )  # 是否在输入前添加 JPEG 压缩作为预处理防御
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=75,
    )  # JPEG 压缩质量（数值越低压缩越强）
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, model_type=args.model_type)
    transform = get_transform()

    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise SystemExit(f"image_dir not found: {image_dir}")
        image_paths = sorted(
            [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not image_paths:
            raise SystemExit(f"no images found in {image_dir}")
    else:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise SystemExit(f"image not found: {image_path}")
        image_paths = [image_path]

    mean, std = get_mean_std(device, torch.float32)
    if args.model_type == "robust":
        mean = torch.zeros_like(mean)
        std = torch.ones_like(std)

    global _DEFENSE_USE_JPEG, _DEFENSE_JPEG_QUALITY
    _DEFENSE_USE_JPEG = bool(args.use_jpeg)
    _DEFENSE_JPEG_QUALITY = int(args.jpeg_quality)

    if args.calibrate_dir is not None:
        calib_dir = Path(args.calibrate_dir)
        if not calib_dir.is_dir():
            raise SystemExit(f"calibrate_dir not found: {calib_dir}")
        calib_paths = sorted(
            [p for p in calib_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not calib_paths:
            raise SystemExit(f"no images found in calibrate_dir: {calib_dir}")
        detector_threshold = calibrate_feature_threshold(
            model,
            calib_paths,
            device=device,
            transform=transform,
            mean=mean,
            std=std,
            n=int(args.calibrate_n),
            quantile=float(args.calibrate_quantile),
        )
    else:
        if args.detector_threshold is not None:
            detector_threshold = float(args.detector_threshold)
        else:
            detector_threshold = calibrate_feature_threshold(
                model,
                image_paths,
                device=device,
                transform=transform,
                mean=mean,
                std=std,
                n=min(100, len(image_paths)),
                quantile=float(args.calibrate_quantile),
            )

    results = {}
    for attack_name in args.attacks:
        if attack_name not in {"fgsm", "pgd", "cw"}:
            continue
        for eps in args.eps_list:
            key = (attack_name, float(eps))
            stats = {
                "clean_correct": 0,
                "attack_success": 0,
                "defense_preproc_success": 0,
                "detector_flags_clean": 0,
                "detector_flags_adv": 0,
                "detector_attack_success": 0,
                "count": 0,
            }
            for path in image_paths:
                x = load_image(path, device, transform)
                logits_clean = predict(model, x, mean, std)
                y_true = logits_clean.argmax(dim=1)

                eval_stats = evaluate_defenses(
                    model=model,
                    x=x,
                    y_true=y_true,
                    mean=mean,
                    std=std,
                    attack_name=attack_name,
                    eps=float(eps),
                    alpha=float(args.alpha),
                    steps=int(args.steps),
                    cw_c=float(args.cw_c),
                    cw_kappa=float(args.cw_kappa),
                    cw_steps=int(args.cw_steps),
                    cw_lr=float(args.cw_lr),
                    detector_threshold=float(detector_threshold),
                )

                for k in stats:
                    if k == "count":
                        continue
                    stats[k] += int(eval_stats[k])
                stats["count"] += 1

            results[key] = stats

    for (attack_name, eps), stats in sorted(results.items()):
        count = max(1, stats["count"])
        attack_success_rate = stats["attack_success"] / count
        preproc_acc = stats["defense_preproc_success"] / count
        detector_clean_rate = 1.0 - stats["detector_flags_clean"] / count
        detector_adv_flag_rate = stats["detector_flags_adv"] / count
        detector_attack_success_rate = stats["detector_attack_success"] / count

        print(
            f"attack={attack_name}, eps={eps:.5f}, "
            f"attack_success={attack_success_rate:.3f}, "
            f"preproc_defense_acc={preproc_acc:.3f}, "
            f"detector_clean_pass_rate={detector_clean_rate:.3f}, "
            f"detector_adv_flag_rate={detector_adv_flag_rate:.3f}, "
            f"detector_attack_success={detector_attack_success_rate:.3f}"
        )


if __name__ == "__main__":
    main()
