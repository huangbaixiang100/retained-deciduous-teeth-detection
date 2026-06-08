#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot ROC and PR curves on external Dataset 2."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms


# ====================  ====================
from config import PROJECT_ROOT, RESULTS_DIR, CROPPED_DIR, DATASET2_SUBDIR, CLASSIFIER_MODEL_PATH, MOUTH_MODEL_PATH, POS_CLASS_NAME, NEG_CLASS_NAMES, ORIGINAL_DIR, NEW_NEG_DIR, NEW_POS_DIR, DATASET1_SUBDIR
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = RESULTS_DIR
from config import CROPPED_DIR  # noqa: F811
DATASET2_SUBDIR = "dataset2"

POS_CLASS_NAME = "retained"
NEG_CLASS_NAMES = ["other_conditions", "normal"]

CLASSIFIER_WEIGHT_PATH = CLASSIFIER_MODEL_PATH

RESNET_IMAGE_SIZE = 224
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_images(dir_path: Path):
    files = []
    if not dir_path.exists():
        return files
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
        files.extend(dir_path.glob(ext))
    return files


def load_dataset2():
    """ (image_paths, labels)label 1=0="""
    base = CROPPED_DIR / DATASET2_SUBDIR
    paths = []
    labels = []
    for p in collect_images(base / POS_CLASS_NAME):
        paths.append(p)
        labels.append(1)
    for neg_class in NEG_CLASS_NAMES:
        for p in collect_images(base / neg_class):
            paths.append(p)
            labels.append(0)
    return paths, labels


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def main():
    print(f": {DEVICE}")
    print(f": {CLASSIFIER_WEIGHT_PATH}")

    paths, labels = load_dataset2()
    if len(paths) == 0:
        raise RuntimeError(
 f"Dataset 2 no imagesrun first complete_pipeline.py complete Stage 1 cropping"
        )
    y_true = np.array(labels)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    print(f"Dataset 2 : {len(paths)}  (={n_pos}, ={n_neg})\n")

    model = build_resnet34(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    transform = get_val_transform()
    dataset = ImagePathDataset(paths, labels, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    #
    y_score = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            y_score.extend(probs[:, 1].cpu().numpy().tolist())
    y_score = np.array(y_score)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Dataset 2)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR
    axes[1].plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {ap:.3f})")
    baseline = n_pos / (n_pos + n_neg)
    axes[1].axhline(y=baseline, color="navy", lw=1, linestyle="--", label=f"No Skill (AP = {baseline:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Dataset 2)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = OUTPUT_DIR / "dataset2_roc_curve.png"
    pr_path = OUTPUT_DIR / "dataset2_pr_curve.png"
    fig.savefig(OUTPUT_DIR / "dataset2_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    #
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve (Dataset 2)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    fig1.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
    ax2.axhline(y=baseline, color="navy", lw=1, linestyle="--", label=f"No Skill ({baseline:.3f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (Dataset 2)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig2.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()

    #  JSON
    metrics = {
        "dataset2_stats": {"total": len(paths), "positive": n_pos, "negative": n_neg},
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }
    with open(OUTPUT_DIR / "dataset2_roc_pr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Dataset 2 ")
    print("=" * 60)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"\n:")
    print(f"  - : {OUTPUT_DIR / 'dataset2_roc_pr_curves.png'}")
    print(f"  - ROC:   {roc_path}")
    print(f"  - PR:    {pr_path}")
    print(f"  - :  {OUTPUT_DIR / 'dataset2_roc_pr_metrics.json'}")


if __name__ == "__main__":
    main()
