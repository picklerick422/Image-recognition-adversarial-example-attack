import argparse
from pathlib import Path

import torch
import torchvision.models as models  # 这里面包含了官方实现的 ResNet 等模型
import torchvision.transforms as transforms  # 图像预处理与数据增强工具
from PIL import Image  #

from attack import cw_l2_attack, fgsm_attack, normalize_batch, pgd_linf_attack

# 加载预训练 ResNet50（设为 eval 模式）
# pretrained=True 表示加载在 ImageNet 上训练好的权重，而不是随机初始化
model = models.resnet50(pretrained=True).eval()

# 根据当前环境自动选择计算设备：
# 如果有可用 GPU（CUDA），则使用 GPU；否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  

# ImageNet 标准化参数（必须使用！）
# mean/std 是在 ImageNet 训练集上统计得到的通道均值和标准差，用来对输入做归一化，
# 使得测试图像与训练时的数据分布一致，从而提升模型表现
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# 定义对输入图片的预处理流程：
transform = transforms.Compose([
    transforms.Resize(256),       # 将较短边缩放到 256 像素，保持长宽比不变
    transforms.CenterCrop(224),  # 从中心裁剪出 224x224 的区域（ResNet 期望的输入大小）
    transforms.ToTensor(),       # 将 PIL 图片转为张量，并把像素值缩放到 [0,1]
])

parser = argparse.ArgumentParser()  # 创建命令行参数解析器
parser.add_argument("image", nargs="?", default="example.jpg")  # 输入图片路径（或目录），默认使用 example.jpg
parser.add_argument("--topk", type=int, default=5)  # 输出前 top-k 个类别
parser.add_argument("--attack", choices=["none", "fgsm", "pgd", "cw"], default="none")  # 选择对抗攻击方式
parser.add_argument("--label", type=int, default=None)  # 指定真实标签（不指定则使用模型预测标签）
parser.add_argument("--eps", type=float, default=8 / 255)  # FGSM/PGD 的扰动大小 ε
parser.add_argument("--alpha", type=float, default=2 / 255)  # PGD 每一步的步长 α
parser.add_argument("--steps", type=int, default=10)  # PGD 迭代步数
parser.add_argument("--cw_c", type=float, default=1.0)  # CW-L2 损失权重系数 c
parser.add_argument("--cw_kappa", type=float, default=0.0)  # CW-L2 置信度参数 κ
parser.add_argument("--cw_steps", type=int, default=1000)  # CW-L2 迭代步数
parser.add_argument("--cw_lr", type=float, default=0.01)  # CW-L2 优化器学习率
parser.add_argument("--target", type=int, default=None)  # CW/PGD 的目标攻击标签（为空则为非定向攻击）
parser.add_argument("--save_adv", type=str, default=None)  # 如指定路径，则保存生成的对抗样本图片
args = parser.parse_args()

categories = None
try:
    from torchvision.models import ResNet50_Weights

    categories = ResNet50_Weights.DEFAULT.meta.get("categories")
except Exception:
    categories = None


def run_for_image(image_path: Path):
    if not image_path.is_file():
        print(f"Image file not found: {image_path}")
        return
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    mean_device = mean.to(device=device, dtype=x.dtype)
    std_device = std.to(device=device, dtype=x.dtype)
    with torch.no_grad():
        logits = model(normalize_batch(x, mean_device, std_device))
    prob = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    topk = max(1, int(args.topk))
    top_prob, top_idx = prob.topk(topk, dim=1)
    top_prob = top_prob[0].cpu().tolist()
    top_idx = top_idx[0].cpu().tolist()
    print(f"Image: {image_path}")
    print("Clean:")
    for rank, (p, idx) in enumerate(zip(top_prob, top_idx), start=1):
        label = categories[idx] if categories and idx < len(categories) else str(idx)
        print(f"Top {rank}: {label} (class {idx}), prob = {p:.4f}")
    if args.attack != "none":
        y_true = torch.tensor([args.label], device=device) if args.label is not None else pred.detach()
        if args.attack == "fgsm":
            x_adv = fgsm_attack(model, x, y_true, eps=float(args.eps), mean=mean_device, std=std_device)
        elif args.attack == "pgd":
            x_adv = pgd_linf_attack(
                model,
                x,
                y_true,
                eps=float(args.eps),
                alpha=float(args.alpha),
                steps=int(args.steps),
                mean=mean_device,
                std=std_device,
            )
        else:
            targeted = args.target is not None
            y_target = torch.tensor([int(args.target)], device=device) if targeted else None
            cw_res = cw_l2_attack(
                model,
                x,
                y_true,
                mean=mean_device,
                std=std_device,
                c=float(args.cw_c),
                kappa=float(args.cw_kappa),
                steps=int(args.cw_steps),
                lr=float(args.cw_lr),
                targeted=targeted,
                y_target=y_target,
            )
            x_adv = cw_res.x_adv
        with torch.no_grad():
            logits_adv = model(normalize_batch(x_adv, mean_device, std_device))
        prob_adv = torch.softmax(logits_adv, dim=1)
        top_prob_adv, top_idx_adv = prob_adv.topk(topk, dim=1)
        top_prob_adv = top_prob_adv[0].cpu().tolist()
        top_idx_adv = top_idx_adv[0].cpu().tolist()
        print(f"Adversarial ({args.attack}):")
        for rank, (p, idx) in enumerate(zip(top_prob_adv, top_idx_adv), start=1):
            label = categories[idx] if categories and idx < len(categories) else str(idx)
            print(f"Top {rank}: {label} (class {idx}), prob = {p:.4f}")
        if args.save_adv:
            out_path = Path(args.save_adv)
            if out_path.parent:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            adv_img = transforms.ToPILImage()(x_adv[0].detach().cpu())
            adv_img.save(out_path)


image_path = Path(args.image)
if image_path.is_dir():
    image_files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(sorted(image_path.glob(pattern)))
    if not image_files:
        raise SystemExit(f"No image files found in directory: {image_path}")
    for img_path in image_files:
        run_for_image(img_path)
else:
    if not image_path.is_file():
        raise SystemExit(
            f"Image file not found: {image_path}. "
            f"Place an image at '{Path('example.jpg').resolve()}' or pass a path: python ResNet.py <image_path>"
        )
    run_for_image(image_path)
