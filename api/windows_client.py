#!/usr/bin/env python3
"""Example Windows client for the RDT Screening API."""

import requests
from pathlib import Path

API_BASE_URL = "http://localhost:15025"


def test_connection():
    print("Testing API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("API connection OK.")
            print(response.json())
            return True
        print(f"Unexpected response: {response.status_code}")
        return False
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def analyze_image(image_path: str):
    print(f"Analyzing image: {image_path}")
    endpoint = f"{API_BASE_URL}/api/v1/analyze"
    path = Path(image_path)
    if not path.exists():
        print(f"File not found: {image_path}")
        return
    with open(path, "rb") as f:
        files = {"file": f}
        response = requests.post(endpoint, files=files, timeout=60)
    if response.status_code == 200:
        result = response.json()
        print("Analysis succeeded.")
        print(f"Classification: {result.get('classification')}")
        print(f"Probability: {result.get('probability')}")
        for rec in result.get("recommendations", []):
            print(f"  - {rec}")
    else:
        print(f"Request failed: HTTP {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    if not test_connection():
        raise SystemExit(1)
    # Set your local test image path here
    analyze_image(r"C:\path\to\test_image.jpg")
