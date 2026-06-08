#!/bin/bash
# Start the RDT Screening API service

echo "======================================"
echo "RDT Screening API"
echo "======================================"

echo "Python version:"
python --version

echo ""
echo "Checking dependencies..."
pip list | grep -E "fastapi|uvicorn|ultralytics|torch"

echo ""
echo "Starting API service..."
echo "URL: http://localhost:15025"
echo "Docs: http://localhost:15025/docs"
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
export API_DEVICE=${API_DEVICE:-cuda:0}
python api_service.py
