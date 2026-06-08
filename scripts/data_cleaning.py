#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset preprocessing script
Clean dataset, standardize labels, and generate dataset statistics.
"""

import os
import json
import shutil
from pathlib import Path
from config import PROJECT_ROOT
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


DISEASE_LABELS = {
    'disease_area',
    'disease area',
    'Disease area',
    'Disease_area'
}

OTHER_DISEASE_LABELS = {
    'other_disease_area',
    'other disease area',
    'Other disease area',
    'Other_disease_area'
}

class DatasetAnalyzer:
    """"""
    def __init__(self):
        self.label_counter = Counter()  
        self.shape_type_counter = Counter()  
        self.points_counter = Counter()  
        self.file_label_counter = defaultdict(Counter)  
        self.label_area_stats = defaultdict(list)  
        self.invalid_labels = []  
        
    def analyze_shape(self, shape, file_path):
        """"""
        label = shape.get('label', '')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])
        
        
        self.label_counter[label] += 1
        self.file_label_counter[file_path][label] += 1
        
        
        self.shape_type_counter[shape_type] += 1
        
        
        if shape_type == 'polygon':
            self.points_counter[len(points)] += 1
        
        
        if len(points) >= 2:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
            self.label_area_stats[label].append(area)
        
        
        if not self._is_valid_label(label):
            self.invalid_labels.append((file_path, label))
    
    def _is_valid_label(self, label):
        """"""
        if label.lower() in [l.lower() for l in DISEASE_LABELS | OTHER_DISEASE_LABELS]:
            return True
        
        if label.replace(',', '').replace('fused_tooth', '').replace(' ', '').isdigit():
            return True
        return False
    
    def generate_report(self, output_dir):
        """"""
        # Path
        invalid_labels = [(str(path), label) for path, label in self.invalid_labels]
        file_label_stats = {str(k): dict(v) for k, v in self.file_label_counter.items()}
        report = {
            'label_stats': dict(self.label_counter),
            'shape_type_stats': dict(self.shape_type_counter),
            'points_stats': dict(self.points_counter),
            'invalid_labels': invalid_labels,
            'file_label_stats': file_label_stats  
        }
        
        
        area_stats = {}
        for label, areas in self.label_area_stats.items():
            if areas:
                area_stats[label] = {
                    'min': min(areas),
                    'max': max(areas),
                    'mean': sum(areas) / len(areas),
                    'count': len(areas)
                }
        report['area_stats'] = area_stats
        
        
        output_dir = Path(output_dir)
        report_path = output_dir / 'dataset_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        
        self._generate_visualizations(output_dir)
        
        return report
    
    def _generate_visualizations(self, output_dir):
        """"""
        output_dir = Path(output_dir)
        
        
        plt.figure(figsize=(12, 8))
        plt.pie(self.label_counter.values(), labels=self.label_counter.keys(), autopct='%1.1f%%')
        plt.title('Label distribution')
        plt.savefig(output_dir / 'label_distribution.png')
        plt.close()
        
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.shape_type_counter.keys(), self.shape_type_counter.values())
        plt.title('Annotation type distribution')
        plt.savefig(output_dir / 'shape_type_distribution.png')
        plt.close()

def normalize_label(label):
    """"""
    
    if label.replace(',', '').replace('fused_tooth', '').replace(' ', '').isdigit():
        return label  
    
    
    label_lower = label.lower()
    if label_lower in [l.lower() for l in DISEASE_LABELS]:
        return 'disease_area'
    if label_lower in [l.lower() for l in OTHER_DISEASE_LABELS]:
        return 'other_disease_area'
    
    return label

def check_labels(json_file_path, stage=1):
    """JSON
    
    Args:
        json_file_path: Path to JSON annotation file
        stage: Processing stage (1=mouth tag check, 2=disease-region check)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        has_mouth = False
        has_disease_area = False
        has_other_disease = False
        
        
        label_counts = {}
        for shape in shapes:
            label = shape.get('label', '').lower()
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # mouth
            if label in ['mouth', 'moyth']:
                has_mouth = True
            
            elif any(l.lower() in label for l in DISEASE_LABELS):
                has_disease_area = True
            elif any(l.lower() in label for l in OTHER_DISEASE_LABELS):
                has_other_disease = True
        
        
        if stage == 1:
            if not has_mouth:
                print(f"\n {json_file_path.name} mouth")
                print("")
                for label, count in label_counts.items():
                    print(f"  - {label}: {count}")
            return has_mouth
        else:  # stage == 2
            if not (has_disease_area or has_other_disease):
                print(f"\n {json_file_path.name} ")
                print("")
                for label, count in label_counts.items():
                    print(f"  - {label}: {count}")
            return has_disease_area or has_other_disease
            
    except Exception as e:
        print(f" {json_file_path} : {e}")
        return False
    except Exception as e:
        print(f" {json_file_path} : {e}")
        return False

def process_json_file(json_file_path, analyzer):
    """JSON"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        modified = False
        
        for shape in shapes:
            
            analyzer.analyze_shape(shape, json_file_path)
            
            
            old_label = shape.get('label', '')
            new_label = normalize_label(old_label)
            if new_label != old_label:
                shape['label'] = new_label
                modified = True
        
        if modified:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f" {json_file_path} : {e}")
        return False

def clean_dataset(source_dir, target_dir, folder_name, analyzer, stage=1):
    """
    
    Args:
        source_dir: Source directory
        target_dir: Target directory
        folder_name: Folder name
        analyzer: Dataset analyzer instance
        stage: Processing stage (1 or 2)
    """
    print(f"\nFolder: {folder_name}")
    
    
    target_folder = Path(target_dir) / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)
    
    json_count = 0
    image_count = 0
    kept_count = 0
    
    # JSON
    source_files = []
    if (Path(source_dir) / "cropped_labels").exists():  
        source_files = list((Path(source_dir) / "cropped_labels").glob('*.json'))
    else:
        source_files = list(Path(source_dir).glob('*.json'))
    for json_path in tqdm(source_files, desc=f"Process {folder_name}"):
        json_count += 1
        
        
        if check_labels(json_path, stage):
            # JSON
            target_json = target_folder / json_path.name
            shutil.copy2(json_path, target_json)
            process_json_file(target_json, analyzer)
            
            
            base_name = json_path.stem
            if base_name.endswith(' jpg'):  
                base_name = base_name[:-4]
            
            
            found = False
            if "cropped_labels" in str(json_path):  
                images_dir = json_path.parent.parent / "cropped_images"
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    image_path = images_dir / f"{base_name}{ext}"
                    if image_path.exists():
                        shutil.copy2(image_path, target_folder / image_path.name)
                        image_count += 1
                        found = True
                        break
            else:  # Original dataset
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    
                    possible_names = [
                        base_name + ext,  
                        base_name.replace(' ', '') + ext,  
                        base_name.replace('_', ' ') + ext,  
                        base_name.replace(' ', '_') + ext,  
                    ]
                    
                    for img_name in possible_names:
                        image_path = json_path.parent / img_name
                        if image_path.exists():
                            shutil.copy2(image_path, target_folder / image_path.name)
                            image_count += 1
                            found = True
                            break
                    
                    if found:
                        break
            
            if not found:
                print(f":  {json_path} ")
            
            kept_count += 1
    
    print(f"{folder_name} :")
    print(f"  - JSON: {json_count}")
    print(f"  - : {kept_count}")
    print(f"  - : {image_count}")
    
    return kept_count

def main():
    """Main"""
    # Original datasetmouth
    base_dir = PROJECT_ROOT / "data/legacy_raw"
    stage1_dir = PROJECT_ROOT / "data/legacy_stage1"
    stats_dir = stage1_dir / "statistics"
    
    print("...")
    print(f": {base_dir}")
    print(f": {stage1_dir}")
    
    
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    
    analyzer = DatasetAnalyzer()
    
    # retained
    ruyazhuliu_dir = base_dir / "retained"
    ruyazhuliu_count = clean_dataset(ruyazhuliu_dir, stage1_dir, "retained", analyzer, stage=1)
    
    # other_conditions
    qitajibing_dir = base_dir / "other_conditions"
    qitajibing_count = clean_dataset(qitajibing_dir, stage1_dir, "other_conditions", analyzer, stage=1)
    
    
    report = analyzer.generate_report(stats_dir)
    
    print("\n!")
    print(f"retained {ruyazhuliu_count} ")
    print(f"other_conditions {qitajibing_count} ")
    print(f" {ruyazhuliu_count + qitajibing_count} ")
    print(f"\n: {stats_dir}")
    
    
    cropped_dir = PROJECT_ROOT / "data/cropped_mouth_dataset"
    stage2_dir = PROJECT_ROOT / "data/legacy_stage2"
    stats_dir = stage2_dir / "statistics"
    
    print("\n...")
    print(f": {cropped_dir}")
    print(f": {stage2_dir}")
    
    
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    
    analyzer = DatasetAnalyzer()
    
    
    cropped_count = clean_dataset(cropped_dir, stage2_dir, "cropped", analyzer, stage=2)
    
    
    report = analyzer.generate_report(stats_dir)
    
    print("\n!")
    print(f"retained {ruyazhuliu_count} ")
    print(f"other_conditions {qitajibing_count} ")
    print(f" {ruyazhuliu_count + qitajibing_count} ")
    print(f"\n: {stats_dir}")
    
    
    print("\n:")
    for label, count in sorted(report['label_stats'].items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {label}: {count}")
    
    
    if report['invalid_labels']:
        print("\n:")
        for file_path, label in report['invalid_labels']:
            print(f"  - {file_path}: {label}")

if __name__ == "__main__":
    main()
