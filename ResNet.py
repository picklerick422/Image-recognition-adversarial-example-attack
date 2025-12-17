import argparse
from pathlib import Path

import torch
import torchvision.models as models  # 这里面包含了官方实现的 ResNet 等模型
import torchvision.transforms as transforms  # 图像预处理与数据增强工具
from PIL import Image  #

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
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# 定义对输入图片的预处理流程：
transform = transforms.Compose([
    transforms.Resize(256),       # 将较短边缩放到 256 像素，保持长宽比不变
    transforms.CenterCrop(224),  # 从中心裁剪出 224x224 的区域（ResNet 期望的输入大小）
    transforms.ToTensor(),       # 将 PIL 图片转为张量，并把像素值缩放到 [0,1]
    normalize,                   # 使用上面的 mean/std 做标准化
])

# 加载一张示例图像（如来自 ImageNet val）
parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", default="example.jpg")
parser.add_argument("--topk", type=int, default=5)
args = parser.parse_args()

image_path = Path(args.image)
if not image_path.is_file():
    raise SystemExit(
        f"Image file not found: {image_path}. "
        f"Place an image at '{Path('example.jpg').resolve()}' or pass a path: python ResNet.py <image_path>"
    )

img = Image.open(image_path).convert("RGB")

# 对图片应用预处理，并增加 batch 维度：
# transform(img) 得到形状 [3, 224, 224] 的张量
# unsqueeze(0) 在最前面增加一个维度，变成 [1, 3, 224, 224]，符合模型输入的(batch_size, C, H, W)
x = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

with torch.no_grad():
    logits = model(x)

prob = torch.softmax(logits, dim=1)

topk = max(1, int(args.topk))
top_prob, top_idx = prob.topk(topk, dim=1)
top_prob = top_prob[0].cpu().tolist()
top_idx = top_idx[0].cpu().tolist()

categories = None
try:
    from torchvision.models import ResNet50_Weights

    categories = ResNet50_Weights.DEFAULT.meta.get("categories")
except Exception:
    categories = None

for rank, (p, idx) in enumerate(zip(top_prob, top_idx), start=1):
    label = categories[idx] if categories and idx < len(categories) else str(idx)
    print(f"Top {rank}: {label} (class {idx}), prob = {p:.4f}")
