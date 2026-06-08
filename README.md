# Automated Detection of Retained Deciduous Teeth from Smartphone Oral Photographs

Implementation of a two-stage deep learning framework (**YOLOv11-small + ResNet-34**) for automated detection of retained deciduous teeth (RDT) from real-world smartphone oral photographs.

## Overview

This repository provides:

- **Stage 1:** YOLOv11-small for oral (mouth) region detection
- **Stage 2:** ResNet-34 for binary classification (retained vs. non-retained deciduous teeth)
- Training, evaluation, robustness analysis, and Grad-CAM visualization scripts
- Pre-trained model weights for inference
- FastAPI-based web service for real-time screening

### Key Results

| Evaluation | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| 5-fold CV (Dataset 1, n=587) | 94.85% | 93.71% | 96.18% | 94.91% |
| External test (Dataset 2, n=526) | 87.73% | 86.43% | **97.30%** | 91.21% |

Stage 1 mouth detection: mAP@0.5 = 0.967

## Repository Structure

```
retained-deciduous-teeth-detection/
├── api/                    # FastAPI inference service
├── configs/                # Training configuration files
├── docs/                   # API documentation
├── models/
│   ├── mouth_detection/    # YOLOv11-small weights (best.pt)
│   └── classifier/         # ResNet-34 weights (resnet34_best_overall.pth)
└── scripts/                # Training and evaluation scripts
```

## Installation

```bash
git clone <repository-url>
cd retained-deciduous-teeth-detection

pip install -r requirements.txt
# For API service:
pip install -r api/requirements_api.txt
```

For GPU inference, install the CUDA-enabled PyTorch build matching your driver version.

## Quick Start: Inference API

```bash
cd api
export API_DEVICE=cuda:0   # or cpu
python api_service.py
```

- API root: http://localhost:15025/
- Interactive docs: http://localhost:15025/docs
- Health check: http://localhost:15025/health

See [docs/API.md](docs/API.md) for detailed endpoint documentation.

## Training Pipeline

Due to privacy restrictions, raw images are not publicly released. To reproduce training, prepare your data under `data/` with the following layout:

```
data/
├── dataset1_raw/
│   ├── retained/
│   ├── other_conditions/
│   └── normal/
├── dataset2_pos/          # external test positives
└── dataset2_neg/          # external test negatives
```

### Stage 1: Mouth Detection

```bash
python scripts/train_mouth_detection.py
```

### Full Two-Stage Pipeline (crop + 5-fold CV + external test)

```bash
python scripts/complete_pipeline.py
```

### Additional Analyses

```bash
python scripts/eval_dataset2_curves.py        # ROC / PR curves
python scripts/noise_robustness_eval.py       # degradation robustness
python scripts/visualize_heatmap.py           # Grad-CAM visualizations
```

## Model Weights

| Model | File | Description |
|---|---|---|
| Mouth detector | `models/mouth_detection/best.pt` | YOLOv11-small, trained for oral region localization |
| RDT classifier | `models/classifier/resnet34_best_overall.pth` | ResNet-34, best overall fold from 5-fold CV |

## Data Availability

Raw image data are not publicly released due to privacy considerations. Access to de-identified data may be granted upon reasonable request to the corresponding author(s) listed in the manuscript.

## Citation

If you use this code or models in your research, please cite the associated manuscript.

## License

This project is released for academic and research purposes.
