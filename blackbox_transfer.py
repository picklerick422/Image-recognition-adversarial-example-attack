import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

from attack import cw_l2_attack, fgsm_attack, normalize_batch, pgd_linf_attack


# 加载用于生成对抗样本的源模型 ResNet50（白盒，支持梯度）
def load_source_model(device: torch.device) -> torch.nn.Module:
    try:
        from torchvision.models import ResNet50_Weights

        model = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    except Exception:
        model = tv_models.resnet50(pretrained=True).eval()
    return model.to(device)


# 加载黑盒目标模型之一：VGG19（传统 CNN 结构）
def load_vgg19(device: torch.device) -> torch.nn.Module:
    try:
        from torchvision.models import VGG19_Weights

        model = tv_models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).eval()
    except Exception:
        model = tv_models.vgg19(pretrained=True).eval()
    return model.to(device)


# 加载黑盒目标模型之二：ViT-B/16（纯 Vision Transformer 结构）
def load_vit(device: torch.device) -> torch.nn.Module:
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    except Exception:
        model = tv_models.vit_b_16(pretrained=True).eval()
    return model.to(device)


# 加载黑盒目标模型之三：Swin-T（层次化、窗口注意力的 Transformer）
def load_swin(device: torch.device) -> torch.nn.Module:
    try:
        from torchvision.models import Swin_T_Weights, swin_t

        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).eval()
    except Exception:
        model = tv_models.swin_t(pretrained=True).eval()
    return model.to(device)


# ImageNet 预训练模型使用的标准化均值和方差
def get_imagenet_mean_std(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
    return mean, std


# 统一的图像预处理：缩放到 256，中心裁剪 224，转为张量
def get_base_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


# 遍历指定文件夹下的所有图片文件
def iter_images(image_dir: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


# 在定模型上做一次前向预测（包含标准化），不计算梯度
def predict_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(normalize_batch(x, mean, std))
    return logits


def main() -> None:
    # 命令行参数：图片目录、攻击类型、攻击强度和可视化数量等
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="picture")
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["fgsm", "pgd", "cw"],
        choices=["fgsm", "pgd", "cw"],
    )
    # FGSM / PGD 的 L∞ 扰动半径 ε（像素范围 [0,1]，8/255≃0.031）
    parser.add_argument("--eps", type=float, default=8 / 255)
    # PGD 每一步的步长 α（建议 α < ε，控制每次更新大小）
    parser.add_argument("--alpha", type=float, default=2 / 255)
    # PGD 的迭代步数（步数越多攻击越强，但计算越慢）
    parser.add_argument("--steps", type=int, default=10)
    # CW-L2 中平衡项 c：权衡扰动大小与分类错误损失
    parser.add_argument("--cw_c", type=float, default=1.0)
    # CW-L2 中置信度参数 κ：要求错误类别比分对的类别大多少（越大越“自信”）
    parser.add_argument("--cw_kappa", type=float, default=0.0)
    # CW-L2 的优化迭代步数（步数越多越容易找到成功对抗样本）
    parser.add_argument("--cw_steps", type=int, default=200)
    # CW-L2 中 Adam 优化器的学习率
    parser.add_argument("--cw_lr", type=float, default=0.01)
    # 为每种攻击最多保存多少组可视化样例（原图 + 对抗图）
    parser.add_argument("--visualize_n", type=int, default=3)
    args = parser.parse_args()

    # 自动选择 CPU 或 GPU 作为运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载源模型和三个黑盒目标模型
    source_model = load_source_model(device)
    vgg19 = load_vgg19(device)
    vit = load_vit(device)
    swin = load_swin(device)

    # 与 ResNet.py 和 defense_experiments.py 保持一致的图像预处理与归一化
    transform = get_base_transform()
    mean, std = get_imagenet_mean_std(device, torch.float32)

    # 统计结构：stats[attack][model] = {"success": 成功次数, "total": 总次数}
    stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))

    # 尝试加载 ImageNet 类别名称，便于可视化时显示标签
    try:
        from torchvision.models import ResNet50_Weights

        categories = ResNet50_Weights.DEFAULT.meta.get("categories")
    except Exception:
        categories = None

    from torchvision.transforms import ToPILImage

    # 如果环境安装了 matplotlib，则开启可视化；否则只做统计
    try:
        import matplotlib.pyplot as plt

        have_matplotlib = True
    except Exception:
        plt = None
        have_matplotlib = False

    to_pil = ToPILImage()
    vis_count = 0  # 已经保存的可视化样本数量

    # 待测试图片所在目录
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise SystemExit(f"image_dir not found: {image_dir}")

    to_pil = ToPILImage()
    vis_count = 0  # 已经保存的可视化样本数量

    # 对 image_dir 下的每一张图片执行：生成对抗样本 + 在黑盒模型上评估迁移性
    for img_path in iter_images(image_dir):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

        # 在源模型 ResNet50 上做干净预测，用于生成对抗样本的“真实标签”
        logits_resnet_clean = predict_model(source_model, x, mean, std)
        y_source = logits_resnet_clean.argmax(dim=1)

        # 在三个黑盒目标模型上做干净图像预测，作为基线标签
        logits_vgg_clean = predict_model(vgg19, x, mean, std)
        y_vgg_clean = logits_vgg_clean.argmax(dim=1)

        logits_vit_clean = predict_model(vit, x, mean, std)
        y_vit_clean = logits_vit_clean.argmax(dim=1)

        logits_swin_clean = predict_model(swin, x, mean, std)
        y_swin_clean = logits_swin_clean.argmax(dim=1)

        # 针对每一种攻击方法，在 ResNet50 上生成对抗样本，再送入黑盒模型评估迁移性
        for attack_name in args.attacks:
            if attack_name == "fgsm":
                # 单步 FGSM 攻击，扰动大小 eps
                x_adv = fgsm_attack(
                    source_model,
                    x,
                    y_source,
                    eps=float(args.eps),
                    mean=mean,
                    std=std,
                )
            elif attack_name == "pgd":
                # 多步 PGD-L∞ 攻击，步长 alpha，迭代 steps 次
                x_adv = pgd_linf_attack(
                    source_model,
                    x,
                    y_source,
                    eps=float(args.eps),
                    alpha=float(args.alpha),
                    steps=int(args.steps),
                    mean=mean,
                    std=std,
                )
            else:
                # CW-L2 攻击（非定向），使用 c、kappa、steps、lr 等参数
                cw_res = cw_l2_attack(
                    source_model,
                    x,
                    y_source,
                    mean=mean,
                    std=std,
                    c=float(args.cw_c),
                    kappa=float(args.cw_kappa),
                    steps=int(args.cw_steps),
                    lr=float(args.cw_lr),
                    targeted=False,
                    y_target=None,
                )
                x_adv = cw_res.x_adv

            # 将对抗样本转为 PIL，便于可视化以及和其它代码复用
            adv_img = to_pil(x_adv[0].detach().cpu())

            # 在三个黑盒目标模型上，对对抗样本做预测
            logits_vgg_adv = predict_model(vgg19, x_adv, mean, std)
            y_vgg_adv = logits_vgg_adv.argmax(dim=1)

            logits_vit_adv = predict_model(vit, x_adv, mean, std)
            y_vit_adv = logits_vit_adv.argmax(dim=1)

            logits_swin_adv = predict_model(swin, x_adv, mean, std)
            y_swin_adv = logits_swin_adv.argmax(dim=1)

            # 如果对抗样本在某个模型上的预测标签与该模型对干净图像的预测不同，则视为迁移攻击成功
            for model_name, y_clean, y_adv in [
                ("VGG19", y_vgg_clean, y_vgg_adv),
                ("ViT", y_vit_clean, y_vit_adv),
                ("Swin", y_swin_clean, y_swin_adv),
            ]:
                stats[attack_name][model_name]["total"] += 1
                if int(y_clean.item()) != int(y_adv.item()):
                    stats[attack_name][model_name]["success"] += 1

            # 可视化部分：保存原图/对抗图 + 各模型预测标签到图片
            if have_matplotlib and vis_count < int(args.visualize_n):
                vis_count += 1

                def idx_to_label(idx: int) -> str:
                    if categories and 0 <= idx < len(categories):
                        return categories[idx]
                    return str(idx)

                # 干净图像下各模型的预测标签
                res_label_clean = idx_to_label(int(y_source.item()))
                vgg_label_clean = idx_to_label(int(y_vgg_clean.item()))
                vit_label_clean = idx_to_label(int(y_vit_clean.item()))
                swin_label_clean = idx_to_label(int(y_swin_clean.item()))

                # 对抗图像下各模型的预测标签
                vgg_label_adv = idx_to_label(int(y_vgg_adv.item()))
                vit_label_adv = idx_to_label(int(y_vit_adv.item()))
                swin_label_adv = idx_to_label(int(y_swin_adv.item()))

                # 左右子图：左为干净输入，右为对抗输入
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(img)
                axes[0].set_title("Clean")
                axes[0].axis("off")

                axes[1].imshow(adv_img)
                axes[1].set_title(f"Adv ({attack_name})")
                axes[1].axis("off")

                # 在图像下方写出各模型的预测标签，便于对比
                clean_text = (
                    f"ResNet: {res_label_clean}\n"
                    f"VGG19: {vgg_label_clean}\n"
                    f"ViT: {vit_label_clean}\n"
                    f"Swin: {swin_label_clean}"
                )
                adv_text = (
                    f"VGG19: {vgg_label_adv}\n"
                    f"ViT: {vit_label_adv}\n"
                    f"Swin: {swin_label_adv}"
                )

                fig.suptitle(f"{img_path.name} ({attack_name})")
                axes[0].text(
                    0.5,
                    -0.1,
                    clean_text,
                    transform=axes[0].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
                axes[1].text(
                    0.5,
                    -0.1,
                    adv_text,
                    transform=axes[1].transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )

                # 输出到 picture/blackbox_vis 目录，文件名包含原图名和攻击方法
                out_dir = image_dir / "blackbox_vis"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{img_path.stem}_{attack_name}.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)

    # 将统计结果汇总为表格形式打印：行=攻击方法，列=目标模型
    models_order = ["VGG19", "ViT", "Swin"]
    attacks_order = ["fgsm", "pgd", "cw"]

    header = ["Attack/Model"] + models_order
    print("\t".join(header))
    for attack_name in attacks_order:
        row = [attack_name.upper()]
        for model_name in models_order:
            total = stats[attack_name][model_name]["total"]
            success = stats[attack_name][model_name]["success"]
            if total == 0:
                asr = 0.0
            else:
                asr = 100.0 * success / total
            row.append(f"{asr:.1f}%")
        print("\t".join(row))


if __name__ == "__main__":
    main()