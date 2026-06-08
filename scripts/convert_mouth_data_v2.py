#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert retained-deciduous-teeth dataset to YOLO format for mouth-detector training.
"""

import os
import json
import shutil
from pathlib import Path
from config import PROJECT_ROOT
from tqdm import tqdm
import random

def convert_to_yolo_format(points, image_width, image_height):
    """YOLO"""
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    
    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height
    width = abs(x2 - x1) / image_width
    height = abs(y2 - y1) / image_height
    
    return center_x, center_y, width, height

def process_json_file(json_path, output_dir, images_dir, labels_dir):
    """JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        image_path = data.get('imagePath', '')
        image_width = data.get('imageWidth', 0)
        image_height = data.get('imageHeight', 0)
        
        if not image_path or not image_width or not image_height:
            return False
        
        
        json_dir = os.path.dirname(json_path)
        full_image_path = os.path.join(json_dir, image_path)
        
        if not os.path.exists(full_image_path):
            print(f": {full_image_path}")
            return False
        
        # mouth
        mouth_annotations = []
        for shape in data.get('shapes', []):
            if shape.get('label', '').lower() == 'mouth' and shape.get('shape_type') == 'rectangle':
                points = shape.get('points', [])
                if len(points) == 2:
                    x, y, w, h = convert_to_yolo_format(points, image_width, image_height)
                    mouth_annotations.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        
        if not mouth_annotations:
            return False
        
        
        base_name = Path(json_path).stem
        new_image_name = f"{base_name}.jpg"
        new_label_name = f"{base_name}.txt"
        
        
        target_image_path = os.path.join(images_dir, new_image_name)
        shutil.copy2(full_image_path, target_image_path)
        
        
        target_label_path = os.path.join(labels_dir, new_label_name)
        with open(target_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(mouth_annotations))
        
        return True
        
    except Exception as e:
        print(f" {json_path} : {e}")
        return False

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """"""
    
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    
    random.shuffle(image_files)
    
    total_count = len(image_files)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    
    def move_files(file_list, split_name):
        for filename in file_list:
            
            src_img = os.path.join(images_dir, filename)
            dst_img = os.path.join(output_dir, split_name, 'images', filename)
            shutil.move(src_img, dst_img)
            
            
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_filename)
            dst_label = os.path.join(output_dir, split_name, 'labels', label_filename)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
    
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')
    
    print(f":")
    print(f": {len(train_files)} ")
    print(f": {len(val_files)} ")
    print(f": {len(test_files)} ")

def main():
    
    source_dir = PROJECT_ROOT / "data/legacy_cleaned"
    output_dir = PROJECT_ROOT / "data/yolo_mouth_dataset_v2"
    
    # Output dir
    temp_images_dir = output_dir / "temp_images"
    temp_labels_dir = output_dir / "temp_labels"
    
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f": {source_dir}")
    print(f"Output dir: {output_dir}")
    
    
    total_files = 0
    processed_files = 0
    
    
    for folder in source_dir.iterdir():
        if folder.is_dir():
            print(f"Folder: {folder.name}")
            
            # JSON
            json_files = list(folder.glob("*.json"))
            total_files += len(json_files)
            
            for json_file in tqdm(json_files, desc=f"Process {folder.name}"):
                if process_json_file(str(json_file), str(output_dir), 
                                   str(temp_images_dir), str(temp_labels_dir)):
                    processed_files += 1
    
    print(f"\n!")
    print(f": {total_files}")
    print(f": {processed_files}")
    print(f"Success rate: {processed_files/total_files*100:.2f}%")
    
    if processed_files > 0:
        print("\n...")
        split_dataset(str(temp_images_dir), str(temp_labels_dir), str(output_dir))
        
        
        shutil.rmtree(temp_images_dir)
        shutil.rmtree(temp_labels_dir)
        
        # YAML
        yaml_content = f"""path: {output_dir}
train: train
val: val
test: test

names:
  0: mouth
"""
        
        yaml_path = output_dir / "mouth_detection_v2.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f": {yaml_path}")
        print("")

if __name__ == "__main__":
    random.seed(42)  
    main()
