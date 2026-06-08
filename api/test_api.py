#!/usr/bin/env python3
"""Simple tests for the RDT Screening API."""

import requests
from pathlib import Path

API_BASE_URL = "http://localhost:15025"


def test_health():
    print("Testing /health")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(response.json())


def test_complete_analysis(image_path: str):
    print(f"Testing /api/v1/analyze with {image_path}")
    with open(image_path, "rb") as f:
        response = requests.post(f"{API_BASE_URL}/api/v1/analyze", files={"file": f})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Classification: {result['classification']}")
        print(f"Probability: {result['probability']}")
    else:
        print(response.text)


if __name__ == "__main__":
    test_health()
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    if sample_dir.exists():
        for img in sample_dir.glob("*.jpg"):
            test_complete_analysis(str(img))
            break
    else:
        print("Place a sample image under data/samples/ to run end-to-end test.")
