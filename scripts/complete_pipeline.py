#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full two-stage pipeline: mouth cropping, 5-fold CV, and external testing."""

import random
import json
import shutil
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torchvision
import torchvision.transforms as transforms

# ==================== Configuration ====================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROJECT_ROOT, ORIGINAL_DIR, NEW_NEG_DIR, NEW_POS_DIR, CROPPED_DIR,
    DATASET1_SUBDIR, DATASET2_SUBDIR, MOUTH_MODEL_PATH, RESULTS_DIR,
    POS_CLASS_NAME, NEG_CLASS_NAMES,
)

BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = RESULTS_DIR

# Train
RESNET_IMAGE_SIZE = 224
N_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOP_START_EPOCH = 16  # Early stopping from epoch 16
EARLY_STOP_PATIENCE = 5  # ValAccuracy5epoch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# negative
          # positive
  # negativeCrop

# Output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Original dataset: {ORIGINAL_DIR}")
print(f"New negatives: {NEW_NEG_DIR}")
print(f"New positives: {NEW_POS_DIR}")
print(f"Cropped dir: {CROPPED_DIR}")
print(f"Output dir: {OUTPUT_DIR}")


# ==================== Utilities ====================

def set_seed(seed=42):
    """"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_images_from_dir(dir_path):
    """"""
    image_files = []
    if not dir_path.exists():
        return image_files
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        image_files.extend(list(dir_path.glob(ext)))
    return image_files


# ==================== Stage 1: MouthCrop ====================

def process_one_source_dir(model, source_dir, target_class_name, target_dataset_subdir, stats, crop_info):
    """Crop images from source_dir using the mouth detector and save to the cropped dataset tree."""
    image_files = collect_images_from_dir(source_dir)
    if not image_files:
        print(f"  Skip (no images): {source_dir}")
        return

    print(f"\nSource: {source_dir}")
    print(f"Dataset/: {target_dataset_subdir}/{target_class_name}")
    output_class_dir = CROPPED_DIR / target_dataset_subdir / target_class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    class_key = f"{target_dataset_subdir}/{target_class_name}"
    if class_key not in stats['by_class']:
        stats['by_class'][class_key] = {'total': 0, 'success': 0, 'failed': 0}

    class_stats = stats['by_class'][class_key]
    class_stats['total'] += len(image_files)

    for img_file in tqdm(image_files, desc=f"Crop {source_dir.name} -> {target_class_name}"):
        stats['total'] += 1
        try:
            results = model(str(img_file), conf=0.25, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                stats['failed'] += 1
                class_stats['failed'] += 1
                continue

            boxes = results[0].boxes
            best_idx = torch.argmax(boxes.conf).item()
            best_box = boxes.xyxy[best_idx].cpu().numpy()

            image = Image.open(img_file).convert('RGB')
            orig_width, orig_height = image.size
            x1, y1, x2, y2 = map(int, best_box)

            
            margin_x = max(int((x2 - x1) * 0.1), 20)
            margin_y = max(int((y2 - y1) * 0.1), 20)

            x1_crop = max(0, x1 - margin_x)
            y1_crop = max(0, y1 - margin_y)
            x2_crop = min(image.width, x2 + margin_x)
            y2_crop = min(image.height, y2 + margin_y)

            if x2_crop > x1_crop and y2_crop > y1_crop:
                cropped_image = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
                output_path = output_class_dir / img_file.name
                cropped_image.save(output_path, quality=95)

                crop_info_key = f"{target_dataset_subdir}/{target_class_name}/{img_file.name}"
                crop_info[crop_info_key] = {
                    'original_size': [orig_width, orig_height],
                    'crop_box': [x1_crop, y1_crop, x2_crop, y2_crop],
                    'cropped_size': [x2_crop - x1_crop, y2_crop - y1_crop],
                    'class': target_class_name,
                    'dataset': target_dataset_subdir,
                    'source_dir': str(source_dir)
                }

                stats['success'] += 1
                class_stats['success'] += 1
            else:
                stats['failed'] += 1
                class_stats['failed'] += 1

        except Exception as e:
            print(f"  Error {img_file.name}: {e}")
            stats['failed'] += 1
            class_stats['failed'] += 1

    print(f"  {source_dir.name} -> {target_class_name}: "
          f"success {class_stats['success']}/{class_stats['total']}")


def stage1_crop_mouths():
    """Stage 1: Train mouth Crop+New data"""
    print("\n" + "=" * 80)
    print("Stage 1: Mouth Croplegacy data")
    print("=" * 80)

    if not MOUTH_MODEL_PATH.exists():
        print(f"Error: Mouth : {MOUTH_MODEL_PATH}")
        return False

    # ClearCropdataset1dataset2 
    print(f"\nClearCroplegacy data: {CROPPED_DIR}")
    old_file_count = 0
    for dsub in [DATASET1_SUBDIR, DATASET2_SUBDIR]:
        for class_name in [POS_CLASS_NAME] + NEG_CLASS_NAMES:
            class_dir = CROPPED_DIR / dsub / class_name
            if class_dir.exists():
                old_files = collect_images_from_dir(class_dir)
                old_file_count += len(old_files)
                for img_file in old_files:
                    try:
                        img_file.unlink()
                    except Exception as e:
                        print(f"  Warning: failed {img_file.name}: {e}")
    if old_file_count > 0:
        print(f"Clear {old_file_count} old files")
    else:
        print("CropClear")

    print(f"Load Mouth : {MOUTH_MODEL_PATH}")
    model = YOLO(str(MOUTH_MODEL_PATH))

    stats = {'total': 0, 'success': 0, 'failed': 0, 'by_class': {}}
    crop_info = {}

    # 1) Dataset1（legacy data）-> dataset1
    process_one_source_dir(model, ORIGINAL_DIR / POS_CLASS_NAME, POS_CLASS_NAME, DATASET1_SUBDIR, stats, crop_info)
    process_one_source_dir(model, ORIGINAL_DIR / NEG_CLASS_NAMES[0], 'other_conditions', DATASET1_SUBDIR, stats, crop_info)
    process_one_source_dir(model, ORIGINAL_DIR / NEG_CLASS_NAMES[1], 'normal', DATASET1_SUBDIR, stats, crop_info)

    # 2) Dataset2（New data）-> dataset2：positive -> retained；negative -> other_conditions
    process_one_source_dir(model, NEW_POS_DIR, POS_CLASS_NAME, DATASET2_SUBDIR, stats, crop_info)
    process_one_source_dir(model, NEW_NEG_DIR, NEG_CLASS_NAMES[0], DATASET2_SUBDIR, stats, crop_info)

    print("\nStage 1complete!")
    print(f"Total images: {stats['total']}")
    print(f"successCrop: {stats['success']}")
    print(f"failed: {stats['failed']}")
    if stats['total'] > 0:
        print(f"success: {stats['success'] / stats['total'] * 100:.2f}%")

    # Crop
    stats['crop_coordinates'] = crop_info
    with open(CROPPED_DIR / 'crop_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats['success'] > 0


# ==================== Stage 2: ResNet34 5Val ====================

class RetentionDataset(Dataset):
    """Binary RDT classification dataset (label 1=retained, 0=non_retained)."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def _load_cropped_from_subdir(cropped_base, dataset_subdir):
    """Load image paths and labels from a cropped dataset subdirectory."""
    base = cropped_base / dataset_subdir
    pos_images = []
    neg_images = []
    stats = {POS_CLASS_NAME: 0, 'negative_classes': {}}

    pos_dir = base / POS_CLASS_NAME
    if pos_dir.exists():
        for img_file in collect_images_from_dir(pos_dir):
            pos_images.append(img_file)
        stats[POS_CLASS_NAME] = len(pos_images)

    for neg_class in NEG_CLASS_NAMES:
        neg_dir = base / neg_class
        count = 0
        if neg_dir.exists():
            for img_file in collect_images_from_dir(neg_dir):
                neg_images.append(img_file)
                count += 1
        stats['negative_classes'][neg_class] = count
    stats['negative_total'] = len(neg_images)
    stats['total_samples'] = len(pos_images) + len(neg_images)
    return pos_images, neg_images, stats


def load_dataset1_with_balance():
    """Load Dataset 1 with class-balanced oversampling for 5-fold CV."""
    pos_images, neg_images, stats = _load_cropped_from_subdir(CROPPED_DIR, DATASET1_SUBDIR)
    pos_count, neg_count = len(pos_images), len(neg_images)

    if neg_count < pos_count:
        n_oversample = pos_count - neg_count
        random.seed(42)
        neg_images = list(neg_images) + random.choices(neg_images, k=n_oversample)
    elif pos_count < neg_count:
        n_oversample = neg_count - pos_count
        random.seed(42)
        pos_images = list(pos_images) + random.choices(pos_images, k=n_oversample)

    all_images = pos_images + neg_images
    all_labels = [1] * len(pos_images) + [0] * len(neg_images)
    stats['oversampled_positive'] = len(pos_images)
    stats['oversampled_negative'] = len(neg_images)
    stats['oversampled_total'] = len(all_images)
    return all_images, all_labels, stats


def load_dataset2():
    """Load Dataset 2 for external testing without resampling."""
    pos_images, neg_images, stats = _load_cropped_from_subdir(CROPPED_DIR, DATASET2_SUBDIR)
    all_images = pos_images + neg_images
    all_labels = [1] * len(pos_images) + [0] * len(neg_images)
    return all_images, all_labels, stats


def print_dataset_stats(stats1, stats2):
    """Dataset1Dataset2"""
    print("\n" + "=" * 60)
    print("DatasetCrop")
    print("=" * 60)
    print("Dataset1legacy data 5 Val")
    print(f"  retained(positive): {stats1.get(POS_CLASS_NAME, 0)} ")
    for k, v in stats1.get('negative_classes', {}).items():
        print(f"  {k}: {v} ")
    print(f"  negative: {stats1.get('negative_total', 0)} ")
    print(f"  total: {stats1.get('total_samples', 0)} ")
    if 'oversampled_total' in stats1:
        print(f"  oversampledtotal: {stats1['oversampled_total']} : {stats1['oversampled_positive']}, : {stats1['oversampled_negative']}")

    print("\nDataset2New dataTest")
    print(f"  retained(positive): {stats2.get(POS_CLASS_NAME, 0)} ")
    for k, v in stats2.get('negative_classes', {}).items():
        print(f"  {k}: {v} ")
    print(f"  negative: {stats2.get('negative_total', 0)} ")
    print(f"  total: {stats2.get('total_samples', 0)} ")
    print("=" * 60)


def get_transforms():
    """Train/Val"""
    train_transform = transforms.Compose([
        transforms.Resize((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


def build_resnet34(num_classes=2):
    """Train ResNet34"""
    try:
        from torchvision.models import resnet34, ResNet34_Weights
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    except Exception:
        #  torchvision
        model = torchvision.models.resnet34(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate_on_test_set(weight_path, test_paths, test_labels, val_transform):
    """Evaluate saved classifier weights on the test set."""
    model = build_resnet34(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    test_dataset = RetentionDataset(test_paths, test_labels, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    # 0=negative, 1=positiveaverage='binary' positive
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        'test_accuracy': float(acc),
        'test_precision': float(prec),
        'test_recall': float(rec),
        'test_f1': float(f1),
        'confusion_matrix': cm,
        'test_samples': int(len(all_labels)),
    }


def train_one_fold(fold_idx, train_paths, train_labels, val_paths, val_labels,
                   train_transform, val_transform):
    """Train one cross-validation fold and return best validation metrics."""
    print("\n" + "=" * 80)
    print(f"Fold {fold_idx + 1}/{N_FOLDS}")
    print("=" * 80)
    print(f"Train: {len(train_paths)}")
    print(f"Val: {len(val_paths)}")
    print(f"Trainpositive: {sum(train_labels)}, negative: {len(train_labels) - sum(train_labels)}")
    print(f"Valpositive: {sum(val_labels)}, negative: {len(val_labels) - sum(val_labels)}")

    train_dataset = RetentionDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = RetentionDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = build_resnet34(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_val_prec = 0.0
    best_val_rec = 0.0
    best_val_f1 = 0.0
    best_state_dict = None
    epochs_without_improvement = 0  

    fold_model_dir = OUTPUT_DIR / 'resnet34_kfold'
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    fold_weight_path = fold_model_dir / f'resnet34_fold{fold_idx + 1}_best.pth'

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Fold {fold_idx+1} Train Epoch {epoch}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / (train_total + 1e-8)
        train_acc = train_correct / (train_total + 1e-8)

        # ValPrecision/Recall/F1
        model.eval()
        val_loss_sum = 0.0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Fold {fold_idx+1} Val Epoch {epoch}/{EPOCHS}"):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_all_preds.extend(preds.cpu().numpy().tolist())
                val_all_labels.extend(labels.cpu().numpy().tolist())

        val_total = len(val_all_labels)
        val_loss = val_loss_sum / (val_total + 1e-8)
        val_acc = (np.array(val_all_preds) == np.array(val_all_labels)).sum() / (val_total + 1e-8)
        val_prec = precision_score(val_all_labels, val_all_preds, average='binary', zero_division=0)
        val_rec = recall_score(val_all_labels, val_all_preds, average='binary', zero_division=0)
        val_f1 = f1_score(val_all_labels, val_all_preds, average='binary', zero_division=0)

        scheduler.step(val_acc)

        print(f"[Fold {fold_idx+1}][Epoch {epoch}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val P: {val_prec:.4f}, R: {val_rec:.4f}, F1: {val_f1:.4f}")

        #  fold Best weightsVal
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_prec = val_prec
            best_val_rec = val_rec
            best_val_f1 = val_f1
            best_state_dict = deepcopy(model.state_dict())
            torch.save(best_state_dict, fold_weight_path)
            epochs_without_improvement = 0  
            print(f"  -> ValAccuracy {fold_weight_path}")
        else:
            # ValAccuracy
            epochs_without_improvement += 1
            if epoch >= EARLY_STOP_START_EPOCH:
                print(f"  -> ValAccuracy ({epochs_without_improvement}/{EARLY_STOP_PATIENCE} epochs)")
        
        # 16epoch
        if epoch >= EARLY_STOP_START_EPOCH and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n  ⚠️  ValAccuracy {EARLY_STOP_PATIENCE} epoch")
            print(f"   {epoch} epochTrainValAccuracy: {best_val_acc:.4f}")
            break

    print(f"\nFold {fold_idx+1} Traincomplete {epoch} epochVal: Acc={best_val_acc:.4f}, P={best_val_prec:.4f}, R={best_val_rec:.4f}, F1={best_val_f1:.4f}")
    return best_val_acc, best_val_prec, best_val_rec, best_val_f1, str(fold_weight_path)


def stage2_train_resnet34_kfold():
    """Run Stage-2 ResNet34 training (5-fold CV) and external testing."""
    print("\n" + "=" * 80)
    print("Stage 2: ResNet34  - Dataset1  5 ValDataset2 Test")
    print("=" * 80)

    # LoadDataset1Train/ValDataset2Test
    train_images, train_labels, stats1 = load_dataset1_with_balance()
    test_images, test_labels, stats2 = load_dataset2()

    print_dataset_stats(stats1, stats2)

    if len(train_images) == 0:
        print("ErrorDataset1 no imagesTrain")
        return

    train_transform, val_transform = get_transforms()

    all_images = np.array(train_images)
    all_labels = np.array(train_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        train_paths = all_images[tr_idx].tolist()
        train_l = all_labels[tr_idx].tolist()
        val_paths = all_images[val_idx].tolist()
        val_l = all_labels[val_idx].tolist()

        best_val_acc, best_val_prec, best_val_rec, best_val_f1, fold_weight_path = train_one_fold(
            fold_idx, train_paths, train_l, val_paths, val_l,
            train_transform, val_transform
        )
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_accuracy': float(best_val_acc),
            'best_val_precision': float(best_val_prec),
            'best_val_recall': float(best_val_rec),
            'best_val_f1': float(best_val_f1),
            'weight_path': fold_weight_path
        })

    # ValAccuracy
    best_fold = max(fold_results, key=lambda x: x['best_val_accuracy'])
    best_overall_path = OUTPUT_DIR / 'resnet34_best_overall.pth'
    shutil.copy2(best_fold['weight_path'], best_overall_path)

    print("\n" + "=" * 80)
    print("5 ValDataset1Val")
    print("=" * 80)
    for r in fold_results:
        print(f"  Fold {r['fold']}: Accuracy={r['best_val_accuracy']:.4f}, "
              f"Precision={r['best_val_precision']:.4f}, Recall={r['best_val_recall']:.4f}, F1={r['best_val_f1']:.4f}")
    mean_cv_acc = np.mean([r['best_val_accuracy'] for r in fold_results])
    mean_cv_prec = np.mean([r['best_val_precision'] for r in fold_results])
    mean_cv_rec = np.mean([r['best_val_recall'] for r in fold_results])
    mean_cv_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    print(f"  Mean: Accuracy={mean_cv_acc:.4f}, Precision={mean_cv_prec:.4f}, Recall={mean_cv_rec:.4f}, F1={mean_cv_f1:.4f}")
    print(f"  Best fold: Fold {best_fold['fold']}, ValAccuracy = {best_fold['best_val_accuracy']:.4f}")
    print(f"  Best weights: {best_overall_path}")

    # Dataset2New dataTest
    test_results = None
    if len(test_images) > 0:
        print("\n" + "=" * 80)
        print("Dataset2New dataTest")
        print("=" * 80)
        test_results = evaluate_on_test_set(
            best_overall_path, test_images, test_labels, val_transform
        )
        print(f"  Test: {test_results['test_samples']}")
        print(f"  TestAccuracy: {test_results['test_accuracy']:.4f}")
        print(f"  TestPrecision(Precision): {test_results['test_precision']:.4f}")
        print(f"  TestRecall(Recall): {test_results['test_recall']:.4f}")
        print(f"  Test F1: {test_results['test_f1']:.4f}")
        print(f"  Confusion matrix(=, =): {test_results['confusion_matrix']}")
    else:
        print("\nDataset2 no imagesSkipTest")

    # Dataset5 Test
    summary = {
        'dataset1_stats': stats1,
        'dataset2_stats': stats2,
        'fold_results': fold_results,
        'mean_cv_accuracy': float(mean_cv_acc),
        'mean_cv_precision': float(mean_cv_prec),
        'mean_cv_recall': float(mean_cv_rec),
        'mean_cv_f1': float(mean_cv_f1),
        'best_fold': best_fold,
        'best_overall_weight': str(best_overall_path),
        'test_results': test_results,
    }
    with open(OUTPUT_DIR / 'resnet34_5fold_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ==================== Main ====================

def main():
    set_seed(42)

    print("\n" + "=" * 80)
    print("Full pipelineMouth Crop + Dataset1  5  CV Train + Dataset2 Test")
    print("=" * 80)

    # Stage 1: Mouth  + Croplegacy data -> dataset1New data -> dataset2
    if not stage1_crop_mouths():
        print("Stage 1failed，exit")
        return

    # Stage 2: Dataset1  5 ValDataset2 Test
    stage2_train_resnet34_kfold()

    print("\n" + "=" * 80)
    print("Full pipelinecomplete！")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("  - Dataset5  CV Test: resnet34_5fold_results.json")
    print("  - Best weights: resnet34_kfold/resnet34_fold{{i}}_best.pth")
    print("  - Best foldTest: resnet34_best_overall.pth")


if __name__ == '__main__':
    main()
