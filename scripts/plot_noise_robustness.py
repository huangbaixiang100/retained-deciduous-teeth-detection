#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot F1 vs. perturbation level from robustness JSON results."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt

#  CJK
try:
    from matplotlib import font_manager
    candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans CN", "SimHei"]
    for f in candidates:
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [f]
            break
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
except Exception:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from config import PROJECT_ROOT, RESULTS_DIR, CROPPED_DIR, DATASET2_SUBDIR, CLASSIFIER_MODEL_PATH, MOUTH_MODEL_PATH, POS_CLASS_NAME, NEG_CLASS_NAMES, ORIGINAL_DIR, NEW_NEG_DIR, NEW_POS_DIR, DATASET1_SUBDIR
BASE_DIR = PROJECT_ROOT
JSON_PATH = BASE_DIR / "resnet34_results" / "dataset2_noise_robustness_results.json"
OUTPUT_DIR = RESULTS_DIR

#
LINE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    settings = data["settings"]
    labels = {
        "gaussian_noise": "Gaussian Noise",
        "salt_pepper": "Salt-Pepper Noise",
        "gaussian_blur": "Gaussian Blur",
        "jpeg_compression": "JPEG Compression",
    }

    levels = list(range(6))
    x_labels = ["0", "1", "2", "3", "4", "5"]

    for idx, (ptype, title) in enumerate(labels.items()):
        if ptype not in settings:
            continue
        s = settings[ptype]
        f1_list = [s[str(l)]["metrics"]["f1"] for l in levels]
        color = LINE_COLORS[idx % len(LINE_COLORS)]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(levels, f1_list, "o-", linewidth=2, markersize=8, color=color)
        ax.set_xticks(levels)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Level")
        ax.set_ylabel("F1-score")
        ax.set_title(title)
        ax.set_ylim(0.88, 0.93)
        ax.axhline(y=f1_list[0], color="gray", linestyle="--", alpha=0.6)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"f1_vs_level_{ptype}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f": {out_path}")

    print("\n4 ")


if __name__ == "__main__":
    main()
