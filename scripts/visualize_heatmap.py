#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grad-CAM and YOLO confidence heatmap visualization."""

import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO

import torchvision


# ====================  ====================
from config import PROJECT_ROOT, RESULTS_DIR, CROPPED_DIR, DATASET2_SUBDIR, CLASSIFIER_MODEL_PATH, MOUTH_MODEL_PATH, POS_CLASS_NAME, NEG_CLASS_NAMES, ORIGINAL_DIR, NEW_NEG_DIR, NEW_POS_DIR, DATASET1_SUBDIR
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = RESULTS_DIR / "heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#
from config import NEW_POS_DIR, NEW_NEG_DIR, MOUTH_MODEL_PATH, CLASSIFIER_MODEL_PATH, RESULTS_DIR   # label=1
   # label=0


CLASSIFIER_WEIGHT_PATH = CLASSIFIER_MODEL_PATH

RESNET_IMAGE_SIZE = 224
CONF_THRESHOLD = 0.25
N_SAMPLES_PER_CLASS = 20 #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "Non-Retention", 1: "Retention"}


# ====================  ====================
def collect_images(dir_path: Path, n: int, seed: int = 42):
    files = []
    if not dir_path.exists():
        return files
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
        files.extend(dir_path.glob(ext))
    random.seed(seed)
    return random.sample(files, min(n, len(files)))


def build_resnet34(num_classes=2):
    try:
        from torchvision.models import resnet34, ResNet34_Weights
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    except Exception:
        model = torchvision.models.resnet34(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ==================== Stage 1: YOLO  +  ====================
def run_yolo_detection(yolo_model, pil_img: Image.Image):
    """
 YOLO :
      best_box: (x1,y1,x2,y2) or None
      conf: float
 conf_heatmap_rgb: HxWx3 uint8 YOLO
    """
    arr = np.array(pil_img)
    results = yolo_model(arr, conf=CONF_THRESHOLD, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, 0.0, None

    boxes = results[0].boxes
    best_idx = torch.argmax(boxes.conf).item()
    best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    conf = float(boxes.conf[best_idx].cpu())

    #
    h, w = arr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        c = float(boxes.conf[i].cpu())
        heatmap[by1:by2, bx1:bx2] = np.maximum(heatmap[by1:by2, bx1:bx2], c)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(arr, 0.5, heatmap_color, 0.5, 0)

    return best_box, conf, blended


def crop_with_margin(pil_img: Image.Image, box):
    x1, y1, x2, y2 = box
    margin_x = max(int((x2 - x1) * 0.1), 20)
    margin_y = max(int((y2 - y1) * 0.1), 20)
    x1c = max(0, x1 - margin_x)
    y1c = max(0, y1 - margin_y)
    x2c = min(pil_img.width, x2 + margin_x)
    y2c = min(pil_img.height, y2 + margin_y)
    return pil_img.crop((x1c, y1c, x2c, y2c))


# ==================== Stage 2: Grad-CAM ====================
class GradCAM:
    """Grad-CAM on ResNet34's last conv block (layer4)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook on the last conv block
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: 1xCxHxW
        Returns cam: HxW float32 in [0,1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # 1xCx1x1
        cam = (weights * self.activations).sum(dim=1).squeeze()    # HxW
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, int(class_idx), float(torch.softmax(logits, dim=1)[0, class_idx].cpu())


def apply_gradcam_overlay(crop_pil: Image.Image, cam: np.ndarray):
    """ Grad-CAM  HxWx3 uint8"""
    cam_resized = cv2.resize(cam, (crop_pil.width, crop_pil.height))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    orig = np.array(crop_pil.convert("RGB"))
    blended = cv2.addWeighted(orig, 0.5, heatmap_color, 0.5, 0)
    return blended


# ====================  ====================
def visualize_sample(yolo_model, resnet_model, gradcam, transform,
                     img_path: Path, gt_label: int, save_path: Path):
    pil_orig = Image.open(img_path).convert("RGB")

    # ---- Stage 1:  ----
    best_box, det_conf, heatmap_blended = run_yolo_detection(yolo_model, pil_orig)

    if best_box is None:
        print(f"  YOLO : {img_path.name}")
        return False

    #
    x1, y1, x2, y2 = best_box
    cv2.rectangle(heatmap_blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(heatmap_blended, f"conf={det_conf:.2f}", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ---- Crop ----
    crop_pil = crop_with_margin(pil_orig, best_box)

    # ---- Stage 2: Grad-CAM ----
    input_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)
    cam, pred_class, pred_conf = gradcam.generate(input_tensor)
    gradcam_img = apply_gradcam_overlay(crop_pil, cam)

    # ----  ----
    gt_name = CLASS_NAMES.get(gt_label, str(gt_label))
    pred_name = CLASS_NAMES.get(pred_class, str(pred_class))
    correct = (pred_class == gt_label)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"{img_path.name}  |  GT: {gt_name}  |  Pred: {pred_name} ({pred_conf:.2%})  "
        f"[{'Correct' if correct else 'Wrong'}]",
        fontsize=12,
    )

    titles = [
        "Original Image",
        "Stage 1: YOLO Detection\n(Confidence Heatmap)",
        "Cropped Mouth Region",
        "Stage 2: ResNet34\n(Grad-CAM)",
    ]
    imgs = [pil_orig, heatmap_blended, crop_pil, gradcam_img]

    for ax, title, img in zip(axes, titles, imgs):
        if isinstance(img, Image.Image):
            ax.imshow(img)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    #
    import matplotlib.cm as cm
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Activation", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  : {save_path.name}")
    return True


# ====================  ====================
def main():
    print(f": {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")

    #
    yolo_model = YOLO(str(MOUTH_MODEL_PATH))
    resnet_model = build_resnet34(num_classes=2).to(DEVICE)
    resnet_model.load_state_dict(
        torch.load(CLASSIFIER_WEIGHT_PATH, map_location=DEVICE)
    )
    resnet_model.eval()
    gradcam = GradCAM(resnet_model)
    transform = get_val_transform()

    #
    pos_files = collect_images(NEW_POS_DIR, N_SAMPLES_PER_CLASS)
    neg_files = collect_images(NEW_NEG_DIR, N_SAMPLES_PER_CLASS)
    samples = [(p, 1) for p in pos_files] + [(p, 0) for p in neg_files]

    print(f"\n: {len(pos_files)} : {len(neg_files)} ")
    print(f" {len(samples)} ...\n")

    ok, skip = 0, 0
    for img_path, label in samples:
        tag = "pos" if label == 1 else "neg"
        save_path = OUTPUT_DIR / f"heatmap_{tag}_{img_path.stem}.png"
        success = visualize_sample(
            yolo_model, resnet_model, gradcam, transform,
            img_path, label, save_path
        )
        if success:
            ok += 1
        else:
            skip += 1

    print(f"\n: {ok}  (YOLO): {skip}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
