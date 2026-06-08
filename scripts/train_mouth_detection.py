#!/usr/bin/env python3
"""Train YOLOv11 mouth region detector (Stage 1)."""

import os
import torch
from ultralytics import YOLO
from config import MOUTH_DETECTION_CONFIG, PROJECT_ROOT
import yaml

def train_mouth_detection():
    """"""

    print("=" * 60)
    print("")
    print("=" * 60)

    # CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # YOLOv11
    model = YOLO('yolo11s.pt')

    #
    config = {
        'data': 'str(PROJECT_ROOT)/mouth_detection.yaml',
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'device': device,
 'patience': 50, #
        'save': True,
        'save_period': 10,
        'cache': True,
        'workers': 4,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'plots': True,
        'val': True,
        'name': 'mouth_detection_yolo11'
    }

    #
    print(":")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    try:
        #
        results = model.train(**config)

        print("\n" + "=" * 60)
        print("")
        print("=" * 60)

        #
        print(":")
        print(f"- mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"- mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"- : {results.results_dict.get('time', 'N/A')}")

        #
        model_save_path = 'str(PROJECT_ROOT)/models/mouth_detection'
        os.makedirs(model_save_path, exist_ok=True)

        #
        best_model_path = os.path.join(model_save_path, 'best.pt')
        model.save(best_model_path)
        print(f": {best_model_path}")

        return True

    except Exception as e:
        print(f": {e}")
        return False

def validate_model():
    """"""

    print("\n" + "=" * 60)
    print("")
    print("=" * 60)

    try:
        #
        model_path = 'str(PROJECT_ROOT)/models/mouth_detection/best.pt'
        if not os.path.exists(model_path):
            print(f": {model_path}")
            return False

        model = YOLO(model_path)

        #
        results = model.val(
            data='str(PROJECT_ROOT)/mouth_detection.yaml',
            split='val',
            save=True,
            plots=True
        )

        print(":")
        print(f"- mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"- mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

        return True

    except Exception as e:
        print(f": {e}")
        return False

if __name__ == "__main__":
    #
    success = train_mouth_detection()

    if success:
        #
        validate_model()

    print("\n" + "=" * 60)
    print("")
    print("")
    print("=" * 60)
