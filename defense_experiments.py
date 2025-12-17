import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from attack import cw_l2_attack, fgsm_attack, normalize_batch, pgd_linf_attack


def load_model(device: torch.device) -> torch.nn.Module:
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


def defense_smoothing(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def defense_quantization(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    x_clamped = x.clamp(0.0, 1.0)
    return torch.round(x_clamped * (levels - 1)) / (levels - 1)


def detector_max_confidence(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    max_conf, _ = prob.max(dim=1)
    return max_conf < threshold


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

    x_adv_smooth = defense_smoothing(x_adv)
    logits_smooth = predict(model, x_adv_smooth, mean, std)
    pred_smooth = logits_smooth.argmax(dim=1)
    defense_smooth_success = int(pred_smooth == y_true)

    x_adv_quant = defense_quantization(x_adv)
    logits_quant = predict(model, x_adv_quant, mean, std)
    pred_quant = logits_quant.argmax(dim=1)
    defense_quant_success = int(pred_quant == y_true)

    detected = detector_max_confidence(logits_adv, detector_threshold)
    detector_flags_adv = int(detected.item())

    detected_clean = detector_max_confidence(logits_clean, detector_threshold)
    detector_flags_clean = int(detected_clean.item())

    detector_attack_success = int((attack_success == 1) and (detector_flags_adv == 0))

    return {
        "clean_correct": clean_correct,
        "attack_success": attack_success,
        "defense_smooth_success": defense_smooth_success,
        "defense_quant_success": defense_quant_success,
        "detector_flags_clean": detector_flags_clean,
        "detector_flags_adv": detector_flags_adv,
        "detector_attack_success": detector_attack_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image", type=str, default="example.jpg")
    parser.add_argument("--attacks", type=str, nargs="+", default=["fgsm", "pgd", "cw"])
    parser.add_argument("--eps_list", type=float, nargs="+", default=[4 / 255, 8 / 255, 16 / 255])
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cw_c", type=float, default=1.0)
    parser.add_argument("--cw_kappa", type=float, default=0.0)
    parser.add_argument("--cw_steps", type=int, default=100)
    parser.add_argument("--cw_lr", type=float, default=0.01)
    parser.add_argument("--detector_threshold", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
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

    results = {}
    for attack_name in args.attacks:
        if attack_name not in {"fgsm", "pgd", "cw"}:
            continue
        for eps in args.eps_list:
            key = (attack_name, float(eps))
            stats = {
                "clean_correct": 0,
                "attack_success": 0,
                "defense_smooth_success": 0,
                "defense_quant_success": 0,
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
                    detector_threshold=float(args.detector_threshold),
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
        smooth_acc = stats["defense_smooth_success"] / count
        quant_acc = stats["defense_quant_success"] / count
        detector_clean_rate = 1.0 - stats["detector_flags_clean"] / count
        detector_adv_flag_rate = stats["detector_flags_adv"] / count
        detector_attack_success_rate = stats["detector_attack_success"] / count

        print(
            f"attack={attack_name}, eps={eps:.5f}, "
            f"attack_success={attack_success_rate:.3f}, "
            f"smooth_defense_acc={smooth_acc:.3f}, "
            f"quant_defense_acc={quant_acc:.3f}, "
            f"detector_clean_pass_rate={detector_clean_rate:.3f}, "
            f"detector_adv_flag_rate={detector_adv_flag_rate:.3f}, "
            f"detector_attack_success={detector_attack_success_rate:.3f}"
        )


if __name__ == "__main__":
    main()

