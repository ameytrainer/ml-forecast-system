#!/bin/bash

# Quick Start Setup Script
# MLOps with Agentic AI - Session 8

echo "=========================================="
echo "ML Forecast System - Quick Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Initialize DVC
echo "Initializing DVC..."
dvc init
echo "✓ DVC initialized"
echo ""

# Configure DVC remote (local for demo)
echo "Configuring DVC remote storage..."
mkdir -p /tmp/dvc-storage
dvc remote add -d localstorage /tmp/dvc-storage
echo "✓ DVC remote configured"
echo ""

# Generate sample data
echo "Generating sample data..."
python scripts/generate_data.py
echo "✓ Sample data generated"
echo ""

# Track data with DVC
echo "Tracking data with DVC..."
dvc add data/raw/sales_data.csv
git add data/raw/sales_data.csv.dvc data/raw/.gitignore
dvc push
echo "✓ Data tracked with DVC"
echo ""

# Preprocess data
echo "Preprocessing data..."
python src/preprocess.py
echo "✓ Data preprocessed"
echo ""

# Train initial model
echo "Training initial model..."
python src/train.py
echo "✓ Model trained"
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start FastAPI backend:"
echo "     uvicorn app.backend:app --reload --port 5000"
echo ""
echo "  2. Start Streamlit dashboard:"
echo "     streamlit run app/dashboard.py"
echo ""
echo "  3. Start MLflow UI:"
echo "     mlflow ui --port 5001"
echo ""
echo "  4. Access applications:"
echo "     - Dashboard: http://localhost:8501"
echo "     - API Docs: http://localhost:5000/docs"
echo "     - MLflow: http://localhost:5001"
echo ""
