#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset 2 noise robustness evaluation (test only).

Uses the same cropped Dataset-2 images as complete_pipeline.py and applies
image degradations before Stage-2 classification.
"""

import io
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from config import PROJECT_ROOT

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torchvision
import torchvision.transforms as transforms


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CROPPED_DIR, DATASET2_SUBDIR, CLASSIFIER_MODEL_PATH, RESULTS_DIR, POS_CLASS_NAME, NEG_CLASS_NAMES

OUTPUT_DIR = RESULTS_DIR
CLASSIFIER_WEIGHT_PATH = CLASSIFIER_MODEL_PATH

RESNET_IMAGE_SIZE = 224
INFER_BATCH_SIZE = 64      #  batch
NUM_WORKERS_PERTURB = 8    
NUM_WORKERS_LOADER = 4     # DataLoader 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Level 0-5 ====================
GAUSSIAN_NOISE_SIGMA = {0: 0,   1: 5,     2: 10,   3: 15,  4: 20,  5: 25}
SALT_PEPPER_P        = {0: 0.0, 1: 0.005, 2: 0.01, 3: 0.02,4: 0.03,5: 0.05}
GAUSSIAN_BLUR_SIGMA  = {0: 0.0, 1: 0.5,   2: 1.0,  3: 1.5, 4: 2.0, 5: 2.5}
JPEG_QUALITY         = {0: 100, 1: 90,    2: 80,   3: 60,  4: 40,  5: 20}


# ==================== Utilities ====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_images_from_dir(dir_path: Path):
    files = []
    if not dir_path.exists():
        return files
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
        files.extend(list(dir_path.glob(ext)))
    return files


def load_dataset2_into_memory():
    """
    Load all cropped Dataset-2 images into memory as numpy arrays.
    Returns a list of {"arr": ndarray, "label": int}.
    """
    base = CROPPED_DIR / DATASET2_SUBDIR
    samples = []

    pos_dir = base / POS_CLASS_NAME
    for p in collect_images_from_dir(pos_dir):
        samples.append({"arr": np.array(Image.open(p).convert("RGB")), "label": 1})

    for neg_class in NEG_CLASS_NAMES:
        neg_dir = base / neg_class
        for p in collect_images_from_dir(neg_dir):
            samples.append({"arr": np.array(Image.open(p).convert("RGB")), "label": 0})

    return samples


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


# ====================  numpy uint8 HxWx3 ====================

def perturb_arr(arr: np.ndarray, perturb_type: str, level: int) -> np.ndarray:
    """numpy uint8 numpy uint8"""
    if perturb_type == "gaussian_noise":
        sigma = GAUSSIAN_NOISE_SIGMA[level]
        if sigma <= 0:
            return arr
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if perturb_type == "salt_pepper":
        prob = SALT_PEPPER_P[level]
        if prob <= 0:
            return arr
        out = arr.copy()
        rnd = np.random.rand(arr.shape[0], arr.shape[1])
        out[rnd < prob / 2] = 0
        out[(rnd >= prob / 2) & (rnd < prob)] = 255
        return out

    if perturb_type == "gaussian_blur":
        sigma = GAUSSIAN_BLUR_SIGMA[level]
        if sigma <= 0:
            return arr
        return np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=sigma)))

    if perturb_type == "jpeg_compression":
        quality = JPEG_QUALITY[level]
        if quality >= 100:
            return arr
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))

    raise ValueError(f"Unknown perturbation: {perturb_type}")


def _perturb_one(args):
    """ ThreadPoolExecutor """
    idx, arr, perturb_type, level = args
    return idx, perturb_arr(arr, perturb_type, level)


# ==================== Dataset numpy  ====================

class PerturbedArrayDataset(Dataset):
    def __init__(self, arrays, labels, transform):
        self.arrays = arrays
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        img = Image.fromarray(self.arrays[idx])
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ====================  +  ====================

def evaluate_one_setting(samples, resnet_model, transform,
                         perturb_type: str, level: int) -> dict:
    """
    Parallel perturbation, batch inference, and metric computation.
    """
    arrs = [s["arr"] for s in samples]
    labels = [s["label"] for s in samples]
    n = len(arrs)

    # Step 1: 
    perturbed = [None] * n
    tasks = [(i, arrs[i], perturb_type, level) for i in range(n)]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS_PERTURB) as pool:
        for idx, result in pool.map(_perturb_one, tasks):
            perturbed[idx] = result

    # Step 2: 
    dataset = PerturbedArrayDataset(perturbed, labels, transform)
    loader = DataLoader(dataset, batch_size=INFER_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS_LOADER, pin_memory=True)

    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            preds = resnet_model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())

    acc  = accuracy_score(labels, all_preds)
    prec = precision_score(labels, all_preds, average="binary", zero_division=0)
    rec  = recall_score(labels, all_preds, average="binary", zero_division=0)
    f1   = f1_score(labels, all_preds, average="binary", zero_division=0)

    return {
        "samples":   n,
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
    }


def level_param_for(perturb_type: str, level: int) -> dict:
    if perturb_type == "gaussian_noise":
        return {"sigma": GAUSSIAN_NOISE_SIGMA[level]}
    if perturb_type == "salt_pepper":
        return {"p": SALT_PEPPER_P[level]}
    if perturb_type == "gaussian_blur":
        return {"sigma": GAUSSIAN_BLUR_SIGMA[level]}
    if perturb_type == "jpeg_compression":
        return {"quality": JPEG_QUALITY[level]}
    return {}


# ==================== Main ====================

def main():
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f": {DEVICE}")
    print(f": {CLASSIFIER_WEIGHT_PATH}")
    print(f": {CROPPED_DIR / DATASET2_SUBDIR}")
    print(":  Stage 1  mouth  Stage 2 ")
    print("      → Level 0  complete_pipeline.py \n")

    if not CLASSIFIER_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Classifier weights missing: {CLASSIFIER_WEIGHT_PATH}")
    if not (CROPPED_DIR / DATASET2_SUBDIR).exists():
        raise FileNotFoundError(
            f"Cropped directory not found: {CROPPED_DIR / DATASET2_SUBDIR}\n"
            "Run complete_pipeline.py Stage 1 first。"
        )

    print(" Dataset 2 ...")
    samples = load_dataset2_into_memory()
    if len(samples) == 0:
        raise RuntimeError("No images in Dataset 2 cropped folder. Run complete_pipeline.py first.")

    pos_count = sum(s["label"] for s in samples)
    neg_count = len(samples) - pos_count
    print(f"Dataset 2 : {len(samples)}  (positive={pos_count}, negative={neg_count})\n")

    # Load model
    resnet_model = build_resnet34(num_classes=2).to(DEVICE)
    resnet_model.load_state_dict(torch.load(CLASSIFIER_WEIGHT_PATH, map_location=DEVICE))
    resnet_model.eval()
    transform = get_val_transform()

    perturb_types = [
        "gaussian_noise",
        "salt_pepper",
        "gaussian_blur",
        "jpeg_compression",
    ]

    all_results = {
        "dataset2_stats": {
            "total": len(samples),
            "positive": int(pos_count),
            "negative": int(neg_count),
        },
        "note": (
            "Perturbations applied to cropped mouth images (CROPPED_DIR/dataset2/). "
            "Stage-2 inference only; consistent with complete_pipeline.py."
        ),
        "settings": {},
    }

    csv_lines = [
        "perturbation,level,param_name,param_value,"
        "samples,accuracy,precision,recall,f1,delta_f1_vs_level0"
    ]

    for ptype in perturb_types:
        print("=" * 70)
        print(f": {ptype}")
        print("=" * 70)
        all_results["settings"][ptype] = {}
        baseline_f1 = None

        for level in range(6):
            params = level_param_for(ptype, level)
            metric = evaluate_one_setting(samples, resnet_model, transform, ptype, level)

            if level == 0:
                baseline_f1 = metric["f1"]
            delta_f1 = metric["f1"] - baseline_f1

            all_results["settings"][ptype][str(level)] = {
                "params": params,
                "metrics": metric,
                "delta_f1_vs_level0": float(delta_f1),
            }

            param_name, param_value = next(iter(params.items()))
            csv_lines.append(
                f"{ptype},{level},{param_name},{param_value},"
                f"{metric['samples']},"
                f"{metric['accuracy']:.6f},{metric['precision']:.6f},"
                f"{metric['recall']:.6f},{metric['f1']:.6f},{delta_f1:.6f}"
            )

            print(
                f"  Level {level} | {param_name}={param_value:<6} | "
                f"F1={metric['f1']:.4f}  Acc={metric['accuracy']:.4f}  "
                f"P={metric['precision']:.4f}  R={metric['recall']:.4f}  "
                f"ΔF1={delta_f1:+.4f}"
            )
        print()

    json_path = OUTPUT_DIR / "dataset2_noise_robustness_results.json"
    csv_path  = OUTPUT_DIR / "dataset2_noise_robustness_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    print("=" * 70)
    print("")
    print("=" * 70)
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")


if __name__ == "__main__":
    main()
