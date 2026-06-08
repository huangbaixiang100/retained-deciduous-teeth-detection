#!/bin/bash
# Stop the RDT Screening API service

echo "Stopping RDT Screening API..."

PID=$(pgrep -f "api_service.py" | head -1)

if [ -z "$PID" ]; then
    echo "No running API service found."
else
    echo "Found process PID=$PID"
    kill "$PID"
    sleep 2
    if pgrep -f "api_service.py" > /dev/null; then
        echo "Process still running; forcing stop..."
        kill -9 "$PID"
    fi
    echo "Service stopped."
fi
