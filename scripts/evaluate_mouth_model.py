#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate YOLOv11 mouth detector
"""

import torch
from ultralytics import YOLO
from pathlib import Path
from config import PROJECT_ROOT
import os
import json
import numpy as np

def evaluate_model(model_path, config_path):
    """"""
    
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    
    if not os.path.exists(config_path):
        print(f"Config missing: {config_path}")
        return None
    
    print(f"Load model: {model_path}")
    model = YOLO(model_path)
    
    print("Evaluate...")
    
    
    results = model.val(data=config_path, split='test')
    
    
    metrics = {
        'precision': float(results.box.mp),  # mAP50-95
        'recall': float(results.box.mr),     # mAP50-95
        'mAP50': float(results.box.map50),   # mAP@0.5
        'mAP50_95': float(results.box.map),  # mAP@0.5:0.95
        'fitness': float(results.fitness),   # Fitness
    }
    
    return metrics, results

def print_metrics(metrics):
    """"""
    print("\n" + "=" * 50)
    print("Evaluation results:")
    print("=" * 50)
    print(f" (Precision): {metrics['precision']:.4f}")
    print(f" (Recall): {metrics['recall']:.4f}")
    print(f"mAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50_95']:.4f}")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print("=" * 50)

def save_metrics(metrics, save_path):
    """JSON"""
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f": {save_path}")
    except Exception as e:
        print(f": {e}")

def main():
    """Main"""
    
    # Model path - 
    model_path = str(PROJECT_ROOT / "runs/detect/mouth_detection_v2/weights/best.pt")
    config_path = str(PROJECT_ROOT / "data/yolo_mouth_dataset_v2/mouth_detection_v2.yaml")
    
    
    results_dir = PROJECT_ROOT / "runs/detect/mouth_detection_v2"
    metrics_file = results_dir / "test_metrics.json"
    
    print("=" * 60)
    print("YOLOv11 mouth model evaluation")
    print("=" * 60)
    
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Train model first!")
        return
    
    
    result = evaluate_model(model_path, config_path)
    
    if result is None:
        print("Evaluation failed!")
        return
    
    metrics, detailed_results = result
    
    
    print_metrics(metrics)
    
    
    results_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, metrics_file)
    
    print(f"\n: {detailed_results.save_dir}")

if __name__ == "__main__":
    main()
