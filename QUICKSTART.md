# 🚀 Quick Start Guide

Get the ML Forecast System running in 5 minutes!

## Prerequisites

- Python 3.10+
- Git
- 4GB RAM minimum

## Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd ml-forecast-system

# Run setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Or on Windows (PowerShell)
python scripts/generate_data.py
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
```

## Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Generate Data

```bash
python scripts/generate_data.py
```

### Step 3: Initialize DVC

```bash
dvc init
mkdir -p /tmp/dvc-storage
dvc remote add -d localstorage /tmp/dvc-storage
dvc add data/raw/sales_data.csv
dvc push
```

### Step 4: Train Model

```bash
# Preprocess data
python src/preprocess.py

# Train model
python src/train.py
```

### Step 5: Start Services

Open 3 terminals:

**Terminal 1 - Backend:**
```bash
uvicorn app.backend:app --reload --port 5000
```

**Terminal 2 - Dashboard:**
```bash
streamlit run app/dashboard.py
```

**Terminal 3 - MLflow:**
```bash
mlflow ui --port 5001
```

## Access Applications

- 📊 **Dashboard**: http://localhost:8501
- 🔌 **API**: http://localhost:5000/docs
- 📈 **MLflow**: http://localhost:5001

## Testing the System

### 1. Make a Prediction via Dashboard

1. Open http://localhost:8501
2. Scroll to "Make a Custom Prediction"
3. Enter values and click "Get Prediction"

### 2. Make a Prediction via API

```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "advertising_spend": 3000,
    "promotions": 1,
    "day_of_week": 0,
    "month": 1,
    "is_weekend": 0
  }'
```

### 3. View in MLflow

1. Open http://localhost:5001
2. Click on "sales-forecaster-dev" experiment
3. View your training run

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Model Not Loading in API

```bash
# Check if model file exists
ls models/trained/model.pkl

# If not, train model
python src/train.py
```

### Dashboard Can't Connect to API

```bash
# Make sure backend is running
curl http://localhost:5000/health

# Start backend if needed
uvicorn app.backend:app --reload --port 5000
```

### DVC Issues

```bash
# Reinitialize DVC
rm -rf .dvc
dvc init
dvc remote add -d localstorage /tmp/dvc-storage
```

## Next Steps

- ✅ Explore the dashboard visualizations
- ✅ Check MLflow experiment tracking
- ✅ Try modifying hyperparameters in `params.yaml`
- ✅ Set up GitHub Actions CI/CD
- ✅ Deploy to cloud (AWS, GCP, Azure)

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review GitHub Issues
- Contact: ml-team@company.com

---

**Happy MLOps! 🚀**
