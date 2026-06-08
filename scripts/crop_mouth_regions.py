#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop mouth regions with trained detector
"""

import torch
from ultralytics import YOLO
from pathlib import Path
from config import PROJECT_ROOT
import os
import json
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

def crop_mouth_regions(model_path, source_dir, output_dir, confidence_threshold=0.25):
    """
    Predict and crop mouth regions
    
    Args:
        model_path: Path to trained detector weights
        source_dir: Source dataset directory
        output_dir: Output dir
        confidence_threshold: Confidence threshold
    """
    
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False
    
    # Load model
    print(f"Load model: {model_path}")
    model = YOLO(model_path)
    
    # Output dir
    output_dir = Path(output_dir)
    cropped_images_dir = output_dir / "cropped_images"
    cropped_labels_dir = output_dir / "cropped_labels"
    
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    cropped_labels_dir.mkdir(parents=True, exist_ok=True)
    
    
    total_images = 0
    successful_crops = 0
    failed_crops = 0
    
    # Source dataset
    source_dir = Path(source_dir)
    
    for folder in source_dir.iterdir():
        if folder.is_dir():
            print(f"Folder: {folder.name}")
            
            # JSON
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            json_files = list(folder.glob("*.json"))
            
            # JSON
            json_map = {}
            for json_file in json_files:
                json_map[json_file.stem] = json_file
            
            for image_file in tqdm(image_files, desc=f"Process {folder.name}"):
                total_images += 1
                
                # JSON
                json_file = json_map.get(image_file.stem)
                if not json_file:
                    print(f"JSON: {image_file}")
                    failed_crops += 1
                    continue
                
                # mouth
                try:
                    results = model(str(image_file), conf=confidence_threshold)
                    
                    if len(results) == 0 or len(results[0].boxes) == 0:
                        print(f"mouth: {image_file}")
                        failed_crops += 1
                        continue
                    
                    
                    boxes = results[0].boxes
                    
                    conf_sorted_idx = torch.argsort(boxes.conf, descending=True)
                    
                    # mouth
                    if len(conf_sorted_idx) > 0:
                        best_idx = conf_sorted_idx[0].item()
                        best_box = boxes.xyxy[best_idx].cpu().numpy()
                        confidence = boxes.conf[best_idx].item()
                        
                        
                        image = Image.open(image_file)
                        x1, y1, x2, y2 = map(int, best_box)
                        
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        margin_x = int(box_width * 0.1)  # 10%
                        margin_y = int(box_height * 0.1)
                        
                        
                        margin_x = max(margin_x, 20)
                        margin_y = max(margin_y, 20)
                        
                        
                        x1 = max(0, x1 - margin_x)
                        y1 = max(0, y1 - margin_y)
                        x2 = min(image.width, x2 + margin_x)
                        y2 = min(image.height, y2 + margin_y)
                        
                        
                        if x2 > x1 and y2 > y1:
                            cropped_image = image.crop((x1, y1, x2, y2))
                            
                            
                            cropped_image_path = cropped_images_dir / f"{folder.name}_{image_file.name}"
                            cropped_image.save(cropped_image_path)
                        else:
                            print(f": {image_file}")
                            failed_crops += 1
                            continue
                    
                    # JSON
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    
                    adjusted_shapes = []
                    for shape in json_data.get('shapes', []):
                        # mouth
                        if shape.get('label', '').lower() in ['mouth', 'moyth']:
                            continue
                        
                        
                        shape_type = shape.get('shape_type', '')
                        points = shape.get('points', [])
                        
                        if shape_type == 'rectangle' and len(points) == 2:
                            
                            new_points = []
                            for point in points:
                                new_x = point[0] - x1
                                new_y = point[1] - y1
                                new_points.append([new_x, new_y])
                            
                            
                            rect_x1 = min(new_points[0][0], new_points[1][0])
                            rect_y1 = min(new_points[0][1], new_points[1][1])
                            rect_x2 = max(new_points[0][0], new_points[1][0])
                            rect_y2 = max(new_points[0][1], new_points[1][1])
                            
                            if (rect_x2 > 0 and rect_x1 < cropped_image.width and
                                rect_y2 > 0 and rect_y1 < cropped_image.height):
                                
                                new_points[0][0] = max(0, min(cropped_image.width, new_points[0][0]))
                                new_points[0][1] = max(0, min(cropped_image.height, new_points[0][1]))
                                new_points[1][0] = max(0, min(cropped_image.width, new_points[1][0]))
                                new_points[1][1] = max(0, min(cropped_image.height, new_points[1][1]))
                                shape['points'] = new_points
                                adjusted_shapes.append(shape)
                        
                        elif shape_type == 'polygon' and len(points) >= 3:
                            
                            new_points = []
                            valid_points = 0
                            points_in_crop = False
                            
                            for point in points:
                                new_x = point[0] - x1
                                new_y = point[1] - y1
                                
                                
                                if (-margin_x <= new_x <= cropped_image.width + margin_x and
                                    -margin_y <= new_y <= cropped_image.height + margin_y):
                                    valid_points += 1
                                
                                
                                if (0 <= new_x <= cropped_image.width and
                                    0 <= new_y <= cropped_image.height):
                                    points_in_crop = True
                                
                                
                                new_x = max(0, min(cropped_image.width, new_x))
                                new_y = max(0, min(cropped_image.height, new_y))
                                new_points.append([new_x, new_y])
                            
                            
                            if points_in_crop and valid_points >= 3:
                                shape['points'] = new_points
                                adjusted_shapes.append(shape)
                    
                    # JSON
                    json_data['shapes'] = adjusted_shapes
                    json_data['imagePath'] = cropped_image_path.name
                    json_data['imageWidth'] = cropped_image.width
                    json_data['imageHeight'] = cropped_image.height
                    
                    
                    json_data['mouth_detection'] = {
                        'confidence': float(confidence),
                        'original_bbox': [float(x) for x in best_box],
                        'crop_region': [x1, y1, x2, y2]
                    }
                    
                    # JSON
                    cropped_json_path = cropped_labels_dir / f"{folder.name}_{json_file.name}"
                    with open(cropped_json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    successful_crops += 1
                    
                except Exception as e:
                    print(f" {image_file} : {e}")
                    failed_crops += 1
    
    
    print(f"\n!")
    print(f"Total images: {total_images}")
    print(f"Cropped: {successful_crops}")
    print(f"Failed: {failed_crops}")
    print(f"Success rate: {successful_crops/total_images*100:.2f}%")
    
    return successful_crops > 0

def main():
    """Main"""
    
    # Model path
    model_path = str(PROJECT_ROOT / "models/mouth_detection/best.pt")
    source_dir = str(PROJECT_ROOT / "data/legacy_stage1")
    output_dir = str(PROJECT_ROOT / "data/cropped_mouth_dataset")
    
    # Confidence threshold
    confidence_threshold = 0.25
    
    print("=" * 60)
    print("Crop mouth regions using trained detector")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Source dataset: {source_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 60)
    
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Train model first!")
        return
    
    
    success = crop_mouth_regions(model_path, source_dir, output_dir, confidence_threshold)
    
    if success:
        print(f"\n! : {output_dir}")
    else:
        print("\n!")

if __name__ == "__main__":
    main()
