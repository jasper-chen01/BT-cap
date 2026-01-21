#!/bin/bash

# Start script for BAT Portal

echo "Starting Brain Tumor Annotation Portal..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if reference embeddings are prepared
if [ ! -f "data/reference_embeddings.faiss" ]; then
    echo "Warning: Reference embeddings not found!"
    echo "Please run: python scripts/prepare_reference_embeddings.py"
    echo "Press Enter to continue anyway, or Ctrl+C to exit..."
    read
fi

# Start backend server
echo "Starting backend server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &

# Wait a moment for server to start
sleep 2

# Start frontend server
echo "Starting frontend server on http://localhost:8080"
cd frontend && python3 -m http.server 8080 &

echo ""
echo "=========================================="
echo "BAT Portal is running!"
echo "Frontend: http://localhost:8080"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
wait
