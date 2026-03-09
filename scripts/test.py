import os
import numpy as np
import cv2
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# -----------------------------
# 项目根目录
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(ROOT, "models")
IMAGE_DIR = os.path.join(ROOT, "dataset/img")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# 用户输入
# -----------------------------
model_name = input("输入模型名称 (例如 sam_vit_b.pth 或 sam_sem_finetune_v1.pth): ")
image_name = input("输入图片名称 (例如 test.tif): ")

model_path = os.path.join(MODEL_DIR, model_name)
image_path = os.path.join(IMAGE_DIR, image_name)

output_path = os.path.join(OUTPUT_DIR, "result.png")


# -----------------------------
# 设备
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# -----------------------------
# 读取图片 (支持16bit tif)
# -----------------------------
def load_image(path):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


image = load_image(image_path)
print("image shape:", image.shape)


# -----------------------------
# 加载 SAM
# -----------------------------
model_type = "vit_b"

print("loading model...")

sam = sam_model_registry[model_type](checkpoint=None)

state_dict = torch.load(model_path, map_location=device)

if isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]

sam.load_state_dict(state_dict)

sam.to(device)
sam.eval()

print("model loaded")


# -----------------------------
# Everything 模式
# -----------------------------
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=100
)

print("running segmentation...")

masks = mask_generator.generate(image)

print("mask number:", len(masks))


# -----------------------------
# 可视化
# -----------------------------
def show_anns(anns, ax):

    if len(anns) == 0:
        return

    for ann in sorted(anns, key=lambda x: x["area"], reverse=True):

        m = ann["segmentation"]

        color = np.concatenate([np.random.random(3), [0.4]])

        overlay = np.zeros((*m.shape, 4))

        overlay[m] = color

        ax.imshow(overlay)


# -----------------------------
# 保存结果
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.imshow(image)

show_anns(masks, ax)

ax.axis("off")

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.close()

print("result saved:", output_path)