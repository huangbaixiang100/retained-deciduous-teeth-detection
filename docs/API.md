# RDT Screening API — External Integration Guide

## Overview

| Item | Value |
|---|---|
| Base URL | `http://localhost:15025` (configure for your deployment) |
| API version | v1.0.0 |
| Protocol | RESTful JSON |
| Authentication | None (configure as needed for production) |

### Quick Start

```python
import requests

BASE_URL = "http://localhost:15025"
response = requests.get(f"{BASE_URL}/health")
print(response.json())
```

---

## Endpoints

### 1. Service Info — `GET /`

Returns service metadata and available endpoints.

**Example response:**
```json
{
  "service": "RDT Screening API",
  "version": "2.0.0",
  "status": "running",
  "device": "cuda:0",
  "endpoints": {
    "complete_analysis": "/api/v1/analyze",
    "mouth_detection": "/api/v1/detect-mouth",
    "classification": "/api/v1/classify",
    "quality_check": "/api/v1/quality-check",
    "heatmap_full": "/api/v1/heatmap",
    "heatmap_stage2": "/api/v1/stage2-heatmap",
    "health": "/health"
  }
}
```

---

### 2. Health Check — `GET /health`

**Example response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda:0",
  "classifier_classes": ["non_retained", "retained"]
}
```

---

### 3. Complete Analysis — `POST /api/v1/analyze` (recommended)

Uploads an oral photograph and runs the full two-stage pipeline.

**Parameters (multipart/form-data):**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Oral photograph |
| `save_image_flag` | bool | No | Save uploaded image on server (default: false) |
| `include_heatmap` | bool | No | Return Stage-2 Grad-CAM heatmap as base64 JPEG (default: false) |

**cURL example:**
```bash
curl -X POST "http://localhost:15025/api/v1/analyze" \
  -F "file=@oral_image.jpg" \
  -F "save_image_flag=false" \
  -F "include_heatmap=true"
```

**Example response:**
```json
{
  "success": true,
  "message": "Analysis complete",
  "mouth_detected": true,
  "classification": "retained",
  "probability": 0.92,
  "probabilities": {
    "non_retained": 0.08,
    "retained": 0.92
  },
  "recommendations": [
    "Retained deciduous teeth detected. Please consult a pediatric dentist."
  ],
  "sharpness": 245.3,
  "exposure": 128.5
}
```

---

### 4. Mouth Detection — `POST /api/v1/detect-mouth`

Stage 1 only: detects and crops the oral region.

---

### 5. Classification — `POST /api/v1/classify`

Stage 2 only: classifies a pre-cropped mouth image.

**Query parameter:** `include_heatmap` (bool) — return Grad-CAM heatmap.

---

### 6. Image Quality Check — `POST /api/v1/quality-check`

Returns sharpness (Laplacian variance) and exposure metrics with quality recommendations.

---

### 7. Two-Stage Heatmap — `POST /api/v1/heatmap`

Returns Stage-1 YOLO confidence heatmap, Stage-2 Grad-CAM heatmap, and a four-panel composite image (all base64 JPEG).

---

### 8. Stage-2 Heatmap Only — `POST /api/v1/stage2-heatmap`

Returns Grad-CAM heatmap for the classification stage.

---

## Classification Labels

| Label | Meaning |
|---|---|
| `retained` | Retained deciduous teeth likely present |
| `non_retained` | No retained deciduous teeth detected |

## Error Handling

HTTP 400 — invalid image format  
HTTP 500 — server-side processing error

## Interactive Documentation

When the service is running locally:

- Swagger UI: http://localhost:15025/docs
- ReDoc: http://localhost:15025/redoc
