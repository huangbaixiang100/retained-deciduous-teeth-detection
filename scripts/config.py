"""Shared project paths for training and evaluation scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

MOUTH_MODEL_PATH = MODELS_DIR / "mouth_detection" / "best.pt"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier" / "resnet34_best_overall.pth"
MOUTH_DETECTION_CONFIG = CONFIGS_DIR / "mouth_detection.yaml"

# User-provided data directories (not included in the public release)
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "dataset1_raw"
NEW_NEG_DIR = DATA_DIR / "dataset2_neg"
NEW_POS_DIR = DATA_DIR / "dataset2_pos"
CROPPED_DIR = DATA_DIR / "cropped"
DATASET1_SUBDIR = "dataset1"
DATASET2_SUBDIR = "dataset2"

# Class folder names (positive / negative)
POS_CLASS_NAME = "retained"
NEG_CLASS_NAMES = ["other_conditions", "normal"]
