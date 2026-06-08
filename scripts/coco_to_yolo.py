import json
import os
import shutil
from pathlib import Path
from config import PROJECT_ROOT
from tqdm import tqdm
import cv2
import numpy as np

def get_image_size(image_path):
    """"""
    try:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
        return None, None
    except Exception as e:
        print(f" {image_path}: {e}")
        return None, None

def convert_coco_to_yolo(json_path, output_dir, class_mapping):
    """COCOJSONYOLO"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        
        image_path = json_path.with_suffix('.jpg')
        if not image_path.exists():
            image_path = json_path.with_suffix('.png')

        if not image_path.exists():
            print(f": {image_path}")
            return False

        img_width, img_height = get_image_size(str(image_path))
        if img_width is None or img_height is None:
            print(f": {image_path}")
            return False

        # Output dir
        output_image_path = output_dir / image_path.name
        shutil.copy2(image_path, output_image_path)

        # txt
        txt_filename = image_path.stem + '.txt'
        txt_path = output_dir / txt_filename

        yolo_annotations = []

        shapes = data.get('shapes', [])
        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', '')

            if label not in class_mapping:
                continue

            class_id = class_mapping[label]

            if shape_type == 'rectangle' and len(points) == 2:
                
                x1, y1 = points[0]
                x2, y2 = points[1]

                # YOLO
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            elif shape_type == 'polygon' and len(points) >= 3:
                #  - 
                points_array = np.array(points)
                x_coords = points_array[:, 0]
                y_coords = points_array[:, 1]

                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                # YOLO
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # YOLO
        if yolo_annotations:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_annotations))

        return True

    except Exception as e:
        print(f" {json_path}: {e}")
        return False

def create_class_mapping():
    """"""
    return {
        'mouth': 0,
        'disease_area': 1,
        'other_disease_area': 2
    }

def convert_dataset(input_dir, output_dir, phase_name):
    """"""
    print(f"Convert {phase_name} ...")

    # Output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    
    class_mapping = create_class_mapping()

    
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")

    # JSON
    json_files = list(input_dir.glob("*.json"))
    print(f" {len(json_files)} JSON")

    success_count = 0
    for json_file in tqdm(json_files, desc=f"Convert {phase_name}"):
        if convert_coco_to_yolo(json_file, output_dir, class_mapping):
            success_count += 1

    print(f"{phase_name} : {success_count}/{len(json_files)} ")
    return success_count

def split_dataset(yolo_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """"""
    print("...")

    
    image_files = list(yolo_dir.glob("*.jpg")) + list(yolo_dir.glob("*.png"))
    total_files = len(image_files)

    if total_files == 0:
        print("")
        return

    
    np.random.shuffle(image_files)

    
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    
    train_dir = yolo_dir / 'train'
    val_dir = yolo_dir / 'val'
    test_dir = yolo_dir / 'test'

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    def move_files(file_list, target_dir):
        for img_file in file_list:
            
            shutil.move(str(img_file), str(target_dir / img_file.name))
            
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                shutil.move(str(txt_file), str(target_dir / txt_file.name))

    
    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)

    
    classes_file = yolo_dir / 'classes.txt'
    if classes_file.exists():
        shutil.copy2(classes_file, train_dir / 'classes.txt')
        shutil.copy2(classes_file, val_dir / 'classes.txt')
        shutil.copy2(classes_file, test_dir / 'classes.txt')

    print(":")
    print(f"- : {len(train_files)}  ({train_ratio*100:.1f}%)")
    print(f"- : {len(val_files)}  ({val_ratio*100:.1f}%)")
    print(f"- : {len(test_files)}  ({test_ratio*100:.1f}%)")

def main():
    
    input_base = PROJECT_ROOT / "data/legacy_stage1"
    output_base = PROJECT_ROOT / "data/yolo_dataset"

    # mouth
    print("=" * 60)
    print("mouth")
    print("=" * 60)

    mouth_output = output_base / "mouth_detection"
    mouth_output.mkdir(parents=True, exist_ok=True)

    # retainedother_conditionsmouth
    print("mouth...")
    success_count = 0

    for category in ["retained", "other_conditions"]:
        input_dir = input_base / category
        if input_dir.exists():
            json_files = list(input_dir.glob("*.json"))
            for json_file in tqdm(json_files, desc=f"Process {category}"):
                # mouth
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    has_mouth = any(shape.get('label') == 'mouth' for shape in data.get('shapes', []))
                    if has_mouth:
                        class_mapping = {'mouth': 0}  # mouth
                        if convert_coco_to_yolo(json_file, mouth_output, class_mapping):
                            success_count += 1
                except Exception as e:
                    print(f" {json_file}: {e}")

    print(f"mouth: {success_count} ")

    # disease_area
    print("\n" + "=" * 60)
    print("disease_area")
    print("=" * 60)

    disease_output = output_base / "disease_detection"
    disease_output.mkdir(parents=True, exist_ok=True)

    class_mapping = create_class_mapping()
    success_count = 0

    for category in ["retained", "other_conditions"]:
        input_dir = input_base / category
        if input_dir.exists():
            json_files = list(input_dir.glob("*.json"))
            for json_file in tqdm(json_files, desc=f"Process {category}"):
                if convert_coco_to_yolo(json_file, disease_output, class_mapping):
                    success_count += 1

    print(f"disease_area: {success_count} ")

    
    print("\n" + "=" * 60)
    print("")
    print("=" * 60)

    split_dataset(mouth_output, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    split_dataset(disease_output, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    print("\n" + "=" * 60)
    print("")
    print("=" * 60)

if __name__ == "__main__":
    main()
