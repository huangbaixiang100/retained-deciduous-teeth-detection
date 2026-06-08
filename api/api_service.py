#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDT Screening API Service
Two-stage endpoints:
1. Mouth region detection
2. Disease detection + classification
"""

import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import torch.serialization
import cv2
from datetime import datetime
import time
import matplotlib.pyplot as plt

# Disable proxy
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# ==================== Configuration ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOUTH_MODEL_PATH = PROJECT_ROOT / 'models/mouth_detection/best.pt'
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / 'models/classifier/resnet34_best_overall.pth'
DISEASE_YOLO_MODEL_PATH = PROJECT_ROOT / 'models/disease_detection/best.pt'
YOLO_POSITIVE_CLASSES = ['disease_area', 'retention', 'retained']
SAVE_IMAGE_DIR = Path(os.environ.get('SAVE_IMAGE_DIR', str(PROJECT_ROOT / 'uploads')))
DEVICE = os.environ.get('API_DEVICE', 'cuda:0') if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['non_retained', 'retained']

# ==================== API ====================
class BoundingBox(BaseModel):
    """"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

class MouthDetectionResponse(BaseModel):
    """Mouth"""
    success: bool
    message: str
    mouth_detected: bool
    mouth_box: Optional[BoundingBox] = None
    cropped_image_base64: Optional[str] = None

class ClassificationResponse(BaseModel):
    """"""
    success: bool
    message: str
    classification: str  # "retained" "non_retained"
    probability: float
    probabilities: Dict[str, float]
    stage2_heatmap_base64: Optional[str] = None  # Grad-CAM
    verification_info: Optional[Dict] = None

class CompleteAnalysisResponse(BaseModel):
    """"""
    success: bool
    message: str
    # 1: Mouth
    mouth_detected: bool
    mouth_box: Optional[BoundingBox] = None
    # 2:
    classification: str  # "retained" "non_retained"
    probability: float
    probabilities: Dict[str, float]
    # base64
    cropped_mouth_image_base64: Optional[str] = None
    stage2_heatmap_base64: Optional[str] = None  # Grad-CAM

    recommendations: List[str]

    sharpness: Optional[float] = None
    exposure: Optional[float] = None

    image_saved: Optional[bool] = None
    saved_image_path: Optional[str] = None

    verification_info: Optional[Dict] = None

class ImageQualityResponse(BaseModel):
    """"""
    success: bool
    message: str
    sharpness: float
    exposure: float
    quality_assessment: str
    recommendations: List[str]


class HeatmapResponse(BaseModel):
    """"""
    success: bool
    message: str
    mouth_detected: bool
    classification: Optional[str] = None
    probability: Optional[float] = None
    stage1_heatmap_base64: Optional[str] = None
    stage2_heatmap_base64: Optional[str] = None
    composite_heatmap_base64: Optional[str] = None


class Stage2HeatmapResponse(BaseModel):
    """（ResNet34 Grad-CAM）"""
    success: bool
    message: str
    mouth_detected: bool
    classification: Optional[str] = None
    probability: Optional[float] = None
    stage2_heatmap_base64: Optional[str] = None
    cropped_mouth_image_base64: Optional[str] = None


# ==================== FastAPI ====================
app = FastAPI(
    title="RDT Screening API",
    description="Two-stage RDT screening with YOLOv11 and ResNet-34",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ====================
print("Loading models...")
mouth_model = None
classifier_model = None
disease_yolo_model = None
transform = None
gradcam_generator = None

def create_classifier_model():
    """ResNet34（）"""
    try:
        from torchvision.models import resnet34, ResNet34_Weights
        model = resnet34(weights=None)
    except Exception:
        # torchvision
        model = models.resnet34(pretrained=False)

    # Linear
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model


class GradCAM:
    """ Grad-CAM """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        # ResNet34
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        """
        input_tensor: 1xCxHxW
        : (cam, pred_idx, pred_prob)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = int(torch.argmax(probs).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)      # 1xCx1x1
        cam = (weights * self.activations).sum(dim=1).squeeze()       # HxW
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        pred_prob = float(probs[class_idx].detach().cpu().numpy())
        return cam, class_idx, pred_prob

@app.on_event("startup")
async def load_models():
    """"""
    global mouth_model, classifier_model, disease_yolo_model, transform, gradcam_generator

    try:
        print(f"Loading mouth model: {MOUTH_MODEL_PATH}")
        mouth_model = YOLO(str(MOUTH_MODEL_PATH))
        # YOLO GPU
        mouth_model.to(DEVICE)
        print(f"Mouth: {DEVICE}")

        print(f"Loading classifier: {CLASSIFIER_MODEL_PATH}")
        classifier_model = create_classifier_model()

        # weights_only PyTorch
        # PyTorch 2.8.0 weights_only=True False
        checkpoint = torch.load(str(CLASSIFIER_MODEL_PATH), map_location=DEVICE, weights_only=False)

        # checkpoint model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # checkpoint fold accuracy
            state_dict = checkpoint['model_state_dict']
            print(f"fold {checkpoint.get('fold', 'N/A')}: {checkpoint.get('val_accuracy', 'N/A')}")
        else:

            state_dict = checkpoint

        classifier_model.load_state_dict(state_dict)

        classifier_model = classifier_model.to(DEVICE)
        classifier_model.eval()
        gradcam_generator = GradCAM(classifier_model)

        if DISEASE_YOLO_MODEL_PATH.exists():
            print(f"Loading optional YOLO disease model: {DISEASE_YOLO_MODEL_PATH}")
            disease_yolo_model = YOLO(str(DISEASE_YOLO_MODEL_PATH))
            disease_yolo_model.to(DEVICE)
        else:
            disease_yolo_model = None
            print("Optional disease detector not found")


        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Image save dir
        SAVE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Image save dir: {SAVE_IMAGE_DIR}")

        print(f"All models loaded: {DEVICE}")
    except Exception as e:
        print(f"Model load failed: {e}")
        import traceback
        traceback.print_exc()
        raise

# ==================== ====================
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def image_to_base64(image: Image.Image) -> str:
    """PILbase64"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def generate_stage2_heatmap_base64(image: Image.Image, class_idx: Optional[int] = None) -> Optional[str]:
    """
     Grad-CAM  base64。
    image: mouth（PIL）
    """
    try:
        if gradcam_generator is None:
            return None

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        cam, pred_idx, _ = gradcam_generator.generate(image_tensor, class_idx=class_idx)

        cam_resized = cv2.resize(cam, (image.width, image.height))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        orig = np.array(image.convert("RGB"))
        blended = cv2.addWeighted(orig, 0.5, heatmap_color, 0.5, 0)
        heatmap_img = Image.fromarray(blended)
        return image_to_base64(heatmap_img)
    except Exception as e:
        print(f": {e}")
        return None

def generate_yolo_stage1_heatmap(image: Image.Image) -> tuple:
    """
     Stage 1 YOLO （ visualize_heatmap.py ）。
     mouth_model（conf=0.25），，
     JET 。

    : (best_box_xyxy, det_conf, blended_array_HWC_uint8)
           (None, 0.0, None)
    """
    try:
        arr = np.array(image.convert("RGB"))
        results = mouth_model(image, conf=0.25, verbose=False, device=DEVICE)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0, None

        boxes = results[0].boxes
        best_idx = int(torch.argmax(boxes.conf).item())
        best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        det_conf = float(boxes.conf[best_idx].cpu())

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

        x1, y1, x2, y2 = best_box
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(blended, f"conf={det_conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return best_box, det_conf, blended
    except Exception as e:
        print(f"Stage1: {e}")
        return None, 0.0, None


def generate_full_heatmap_visualization(image: Image.Image) -> tuple:
    """
    （ visualize_heatmap.py ）。

    ：
      1. Stage 1：mouth_model  → YOLO
      2.  mouth （10%）
      3. Stage 2：ResNet34  + Grad-CAM

    :
      (classification, probability,
       stage1_heatmap_base64, stage2_heatmap_base64,
       composite_heatmap_base64, message)
    """
    try:
        # ---- Stage 1 ----
        best_box, det_conf, stage1_arr = generate_yolo_stage1_heatmap(image)
        if best_box is None:
            return None, 0.0, None, None, None, "Oral region not detected"

        # ---- ----
        x1, y1, x2, y2 = best_box
        margin_x = max(int((x2 - x1) * 0.1), 20)
        margin_y = max(int((y2 - y1) * 0.1), 20)
        x1c = max(0, x1 - margin_x)
        y1c = max(0, y1 - margin_y)
        x2c = min(image.width, x2 + margin_x)
        y2c = min(image.height, y2 + margin_y)
        crop_pil = image.crop((x1c, y1c, x2c, y2c))

        # ---- Stage 2: Grad-CAM ----
        input_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)
        cam, pred_class_idx, pred_prob = gradcam_generator.generate(input_tensor)
        classification = CLASS_NAMES[pred_class_idx]

        cam_resized = cv2.resize(cam, (crop_pil.width, crop_pil.height))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        orig_crop = np.array(crop_pil.convert("RGB"))
        stage2_arr = cv2.addWeighted(orig_crop, 0.5, heatmap_color, 0.5, 0)

        # ---- base64 ----
        def arr_to_b64(arr):
            img = Image.fromarray(arr.astype(np.uint8))
            return image_to_base64(img)

        stage1_b64 = arr_to_b64(stage1_arr)
        stage2_b64 = arr_to_b64(stage2_arr)

        # ---- ----
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        fig.suptitle(
            f"Pred: {classification} ({pred_prob:.2%})  |  YOLO conf: {det_conf:.2f}",
            fontsize=12
        )
        panel_titles = [
            "Original Image",
            "Stage 1: YOLO Detection\n(Confidence Heatmap)",
            "Cropped Mouth Region",
            "Stage 2: ResNet34\n(Grad-CAM)",
        ]
        panel_imgs = [image, stage1_arr, crop_pil, stage2_arr]
        for ax, title, img in zip(axes, panel_titles, panel_imgs):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("Activation", fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="JPEG", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        composite_b64 = base64.b64encode(buf.getvalue()).decode()

        return classification, float(pred_prob), stage1_b64, stage2_b64, composite_b64, "success"

    except Exception as e:
        print(f": {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, None, None, None, f"Generation failed: {str(e)}"


def detect_mouth(image: Image.Image) -> tuple:
    """
    mouth
    : (success, cropped_image, box_info, all_detections)
    """
    try:
        results = mouth_model(image, conf=0.7, verbose=False, device=DEVICE)

        if len(results) == 0 or len(results[0].boxes) == 0:
            print("")
            return False, None, None, []


        boxes = results[0].boxes
        all_detections = []


        print("=" * 60)
        print("")
        print("=" * 60)
        for i, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)
            all_detections.append({
                'idx': i,
                'confidence': conf,
                'box': [x1, y1, x2, y2],
                'area': (x2 - x1) * (y2 - y1)
            })
            print(f" {i+1}: ={conf:.4f} | =[{x1}, {y1}, {x2}, {y2}] | ={ (x2-x1)*(y2-y1) }")
        print("=" * 60)


        best_idx = torch.argmax(boxes.conf).item()
        best_box = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(boxes.conf[best_idx].cpu().numpy())
        print(f" #{best_idx+1}: {confidence:.4f}")
        print("=" * 60)


        x1, y1, x2, y2 = map(int, best_box)


        margin_x = max(int((x2 - x1) * 0.1), 20)
        margin_y = max(int((y2 - y1) * 0.1), 20)

        x1_crop = max(0, x1 - margin_x)
        y1_crop = max(0, y1 - margin_y)
        x2_crop = min(image.width, x2 + margin_x)
        y2_crop = min(image.height, y2 + margin_y)

        cropped_image = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))

        box_info = BoundingBox(
            x1=float(x1_crop),
            y1=float(y1_crop),
            x2=float(x2_crop),
            y2=float(y2_crop),
            confidence=confidence,
            class_name="mouth"
        )

        return True, cropped_image, box_info, all_detections

    except Exception as e:
        print(f"Mouth: {e}")
        return False, None, None, []

def detect_disease_with_yolo(image: Image.Image) -> tuple:
    """
    YOLO（）
    : (classification, confidence, detection_info)
    """
    try:
        # YOLO
        results = disease_yolo_model(image, conf=0.25, verbose=False, device=DEVICE)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return "non_retained", 0.0, {"message": "YOLO found no disease"}

        boxes = results[0].boxes


        detections = []
        for i, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = disease_yolo_model.names[cls_id]
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)

            detections.append({
                'idx': i,
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'box': [x1, y1, x2, y2]
            })


        best_detection = max(detections, key=lambda x: x['confidence'])
        confidence = best_detection['confidence']
        class_name = best_detection['class_name']

        # YOLO
        # YOLO_POSITIVE_CLASSES
        if class_name in YOLO_POSITIVE_CLASSES:
            classification = "retained"
        else:
            classification = "non_retained"

        print("=" * 60)
        print(f"YOLO dual verification：")
        print(f"  : {len(detections)}")
        print(f"  : ={class_name}, ={confidence:.4f}")
        print(f"  : {classification}")
        print("=" * 60)

        return classification, confidence, {
            "best_detection": best_detection,
            "all_detections": detections,
            "message": f"YOLO{len(detections)}"
        }

    except Exception as e:
        print(f"YOLO: {e}")
        return "non_retained", 0.0, {"message": f"YOLO detection failed: {str(e)}"}


def classify_image(image: Image.Image, include_heatmap: bool = False) -> tuple:
    """
    mouth（YOLO dual verification）
    : (classification, probability, probabilities_dict, verification_info, stage2_heatmap_base64)
    """
    try:

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)


        with torch.no_grad():
            outputs = classifier_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_prob = probabilities[predicted_class_idx].item()


        probabilities_dict = {
            CLASS_NAMES[i]: float(probabilities[i].cpu().numpy())
            for i in range(len(CLASS_NAMES))
        }

        classification = CLASS_NAMES[predicted_class_idx]
        verification_info = None
        final_probability = predicted_prob  # ResNet

        # 40%-60% YOLO
        if 0.4 <= predicted_prob < 0.6:
            print(f"\n⚠️ ResNet ({predicted_prob:.2%})YOLO dual verification...")

            # YOLO
            yolo_classification, yolo_confidence, yolo_info = detect_disease_with_yolo(image)

            # YOLO
            classification = yolo_classification
            final_probability = yolo_confidence  # YOLO
            verification_info = {
                "method": "yolo_override",
                "resnet_prob": predicted_prob,
                "resnet_class": CLASS_NAMES[predicted_class_idx],
                "yolo_prob": yolo_confidence,
                "yolo_class": yolo_classification,
                "final_class": classification,
                "reason": "YOLO override"
            }
            print(f"  ✅ YOLO override: {classification} (: {yolo_confidence:.2%})")


        stage2_heatmap_base64 = None
        if include_heatmap:
            # YOLO ResNet
            stage2_heatmap_base64 = generate_stage2_heatmap_base64(
                image, class_idx=predicted_class_idx
            )

        return classification, final_probability, probabilities_dict, verification_info, stage2_heatmap_base64

    except Exception as e:
        print(f": {e}")
        return "unknown", 0.0, {}, None, None

def calculate_sharpness(image: Image.Image) -> float:
    """
    （Laplacian Variance）
    ，
    """
    # PIL OpenCV
    img_array = np.array(image)
    # BGR RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array


    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)

def calculate_exposure(image: Image.Image) -> float:
    """
    （）
    ，
    """
    # PIL OpenCV
    img_array = np.array(image)
    # BGR RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array


    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    avg_brightness = np.mean(gray)
    return float(avg_brightness)

def save_image(image: Image.Image, save_dir: Path) -> tuple:
    """

    : (success, saved_path)
    """
    try:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"image_{timestamp}.jpg"
        save_path = save_dir / filename

        # JPEG
        image.save(save_path, format='JPEG', quality=95)

        return True, str(save_path)
    except Exception as e:
        print(f": {e}")
        return False, None

def assess_image_quality(sharpness: float, exposure: float) -> tuple:
    recommendations, quality_issues = [], []
    if sharpness < 100:
        quality_issues.append("blurry"); recommendations.append("Image appears blurry. Retake a sharper photo.")
    elif sharpness < 200:
        quality_issues.append("moderate"); recommendations.append("Image sharpness is moderate.")
    else:
        recommendations.append("Image sharpness is good.")
    if exposure < 50:
        quality_issues.append("under"); recommendations.append("Image is too dark.")
    elif exposure > 200:
        quality_issues.append("over"); recommendations.append("Image is too bright.")
    elif exposure < 80 or exposure > 180:
        quality_issues.append("suboptimal"); recommendations.append("Exposure is suboptimal.")
    else:
        recommendations.append("Exposure is acceptable.")
    quality_level = "excellent" if not quality_issues else ("good" if len(quality_issues)==1 else "needs_improvement")
    return quality_level, recommendations

def generate_recommendations(classification: str, probability: float, verification_info: Optional[Dict] = None) -> List[str]:
    recommendations = []
    if verification_info and verification_info.get("method") == "yolo_override":
        recommendations += ["ResNet confidence uncertain; YOLO dual verification applied.", ""]
    if classification == "retained":
        recommendations += ["Possible retained deciduous teeth detected. Consult a pediatric dentist.",
            "Consider a dental visit within one week.", "Maintain oral hygiene."]
    else:
        if probability < 0.6:
            recommendations += ["Low confidence. Retake photo or consult a dentist."]
        else:
            recommendations += ["No obvious RDT detected in this screening result."]
        recommendations += ["Continue routine oral hygiene."]
    return recommendations


# ==================== API ====================

@app.get("/")
async def root():
    """API"""
    return {
        "service": "RDT Screening API",
        "version": "2.0.0",
        "status": "running",
        "device": DEVICE,
        "save_image_directory": str(SAVE_IMAGE_DIR),
        "endpoints": {
            "complete_analysis": "/api/v1/analyze (save_image_flaginclude_heatmap)",
            "mouth_detection": "/api/v1/detect-mouth",
            "classification": "/api/v1/classify (include_heatmap)",
            "quality_check": "/api/v1/quality-check",
            "heatmap_full": "/api/v1/heatmap ()",
            "heatmap_stage2": "/api/v1/stage2-heatmap (Grad-CAM)",
            "health": "/health"
        },
        "new_features": [
            " (save_image_flag)",
            " ()",
            " ()",
            "",
            "Grad-CAM (include_heatmap / /api/v1/stage2-heatmap)"
        ]
    }

@app.get("/health")
async def health_check():
    """"""
    return {
        "status": "healthy",
        "models_loaded": mouth_model is not None and classifier_model is not None,
        "device": DEVICE,
        "classifier_classes": CLASS_NAMES
    }

@app.post("/api/v1/detect-mouth", response_model=MouthDetectionResponse)
async def detect_mouth_endpoint(file: UploadFile = File(...)):
    """
    1: Mouth region detection

    ，mouth
    """
    try:

        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)

        # mouth
        success, cropped_image, box_info, all_detections = detect_mouth(image)

        if success:
            cropped_base64 = image_to_base64(cropped_image)
            return MouthDetectionResponse(
                success=True,
                message="Mouth region detected successfully",
                mouth_detected=True,
                mouth_box=box_info,
                cropped_image_base64=cropped_base64
            )
        else:
            return MouthDetectionResponse(
                success=False,
                message="Mouth region not detected，Ensure a clear oral cavity is visible",
                mouth_detected=False
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/v1/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    file: UploadFile = File(...),
    include_heatmap: bool = Query(False, description="Grad-CAMbase64")
):
    """
    2: （mouth）

    mouth，。
    （ResNet34）Grad-CAM。
    """
    try:

        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)


        classification, probability, probabilities_dict, verification_info, stage2_heatmap_base64 = classify_image(
            image, include_heatmap=include_heatmap
        )


        message = f"Classification complete: {classification}"
        if verification_info and verification_info["method"] == "yolo_override":
            message += " (YOLO dual verification)"

        return ClassificationResponse(
            success=True,
            message=message,
            classification=classification,
            probability=probability,
            probabilities=probabilities_dict,
            stage2_heatmap_base64=stage2_heatmap_base64,
            verification_info=verification_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/v1/analyze", response_model=CompleteAnalysisResponse)
async def complete_analysis(
    file: UploadFile = File(...),
    save_image_flag: bool = Query(False, description=""),
    include_heatmap: bool = Query(False, description="Grad-CAMbase64")
):
    """
    （）

    ，：
    1. Mouth region detection
    2. （ vs ）

    :
    - file:
    - save_image_flag: （False）
    - include_heatmap: （False）


    """
    try:

        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)


        image_saved = False
        saved_path = None
        if save_image_flag:
            image_saved, saved_path = save_image(image, SAVE_IMAGE_DIR)


        sharpness = calculate_sharpness(image)
        exposure = calculate_exposure(image)

        # 1: Mouth
        success, cropped_image, mouth_box, all_detections = detect_mouth(image)

        if not success:
            return CompleteAnalysisResponse(
                success=False,
                message="Mouth region not detected，Upload an image with a visible oral cavity",
                mouth_detected=False,
                classification="unknown",
                probability=0.0,
                probabilities={},
                recommendations=["📸 Retake photo with clear oral cavity"],
                sharpness=sharpness,
                exposure=exposure,
                image_saved=image_saved,
                saved_image_path=saved_path
            )

        # 2:
        classification, probability, probabilities_dict, verification_info, stage2_heatmap_base64 = classify_image(
            cropped_image, include_heatmap=include_heatmap
        )

        # base64
        cropped_base64 = image_to_base64(cropped_image)


        recommendations = generate_recommendations(classification, probability, verification_info)


        message = "Analysis complete"
        if verification_info and verification_info["method"] == "yolo_override":
            message = "Analysis complete (YOLO dual verification)"

        return CompleteAnalysisResponse(
            success=True,
            message=message,
            mouth_detected=True,
            mouth_box=mouth_box,
            classification=classification,
            probability=probability,
            probabilities=probabilities_dict,
            cropped_mouth_image_base64=cropped_base64,
            stage2_heatmap_base64=stage2_heatmap_base64,
            recommendations=recommendations,
            sharpness=sharpness,
            exposure=exposure,
            image_saved=image_saved,
            saved_image_path=saved_path,
            verification_info=verification_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/v1/heatmap", response_model=HeatmapResponse)
async def heatmap_endpoint(file: UploadFile = File(...)):
    """
    （）

    ，：

    - **Stage 1**： YOLOv11 ，
       JET ，。
    - **Stage 2**： mouth  ResNet34 ， **Grad-CAM**
      ，。

    （ base64  JPEG）：
    - `stage1_heatmap_base64`：Stage 1 YOLO （）
    - `stage2_heatmap_base64`：Stage 2 Grad-CAM （ mouth ）
    - `composite_heatmap_base64`：
      （ | YOLO |  | Grad-CAM），
    """
    try:
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)

        classification, probability, stage1_b64, stage2_b64, composite_b64, msg = \
            generate_full_heatmap_visualization(image)

        if classification is None:
            return HeatmapResponse(
                success=False,
                message=msg,
                mouth_detected=False,
            )

        return HeatmapResponse(
            success=True,
            message=f"Heatmap generated: {classification} ({probability:.2%})",
            mouth_detected=True,
            classification=classification,
            probability=probability,
            stage1_heatmap_base64=stage1_b64,
            stage2_heatmap_base64=stage2_b64,
            composite_heatmap_base64=composite_b64,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/stage2-heatmap", response_model=Stage2HeatmapResponse)
async def stage2_heatmap_endpoint(file: UploadFile = File(...)):
    """
     Grad-CAM

    ， Mouth ， ResNet34 ，
     **Grad-CAM** ，。

    :
    - `stage2_heatmap_base64`:  Grad-CAM （ mouth ），Base64 JPEG
    - `classification`: （ / ）
    - `probability`:
    - `cropped_mouth_image_base64`: （）
    """
    try:
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)

        success, cropped_image, mouth_box, _ = detect_mouth(image)
        if not success:
            return Stage2HeatmapResponse(
                success=False,
                message="Oral region not detected，Cannot generate heatmap",
                mouth_detected=False,
            )

        classification, probability, _, _, stage2_heatmap_base64 = classify_image(
            cropped_image, include_heatmap=True
        )
        cropped_base64 = image_to_base64(cropped_image)

        return Stage2HeatmapResponse(
            success=True,
            message="Heatmap generated",
            mouth_detected=True,
            classification=classification,
            probability=probability,
            stage2_heatmap_base64=stage2_heatmap_base64,
            cropped_mouth_image_base64=cropped_base64,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/quality-check", response_model=ImageQualityResponse)
async def check_image_quality(file: UploadFile = File(...)):
    """


    ，，

    :
    - sharpness: （）
    - exposure: （）
    - quality_assessment:
    - recommendations:
    """
    try:

        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)


        sharpness = calculate_sharpness(image)
        exposure = calculate_exposure(image)


        quality_level, recommendations = assess_image_quality(sharpness, exposure)

        return ImageQualityResponse(
            success=True,
            message=f"Image quality check complete. Level: {quality_level}",
            sharpness=sharpness,
            exposure=exposure,
            quality_assessment=quality_level,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ==================== ====================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("API_PORT", 15025))  # 1001

    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # False
        workers=1
    )

