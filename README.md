# 📊 Sales Forecaster - Production ML System

Complete end-to-end MLOps pipeline with DVC, MLflow, and GitHub Actions CI/CD.

## 🎯 Overview

This is a production-grade ML system that demonstrates:
- ✅ Automated ML pipeline with 8 stages
- ✅ DVC for dataset versioning
- ✅ MLflow for experiment tracking & model registry
- ✅ GitHub Actions for CI/CD automation
- ✅ Multi-environment deployment (Dev/Staging/Prod)
- ✅ Complete traceability (Git + DVC + MLflow)
- ✅ Production dashboard with live predictions
- ✅ Monitoring, alerting & rollback procedures

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- (Optional) GitHub account for CI/CD

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd ml-forecast-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
# Initialize DVC
dvc init

# Configure local remote storage (for demo)
mkdir -p /tmp/dvc-storage
dvc remote add -d localstorage /tmp/dvc-storage

# For production, use cloud storage:
# dvc remote add -d production s3://your-bucket/dvc-store
```

### 3. Generate Sample Data

```bash
# Generate synthetic sales data
python scripts/generate_data.py

# Track data with DVC
dvc add data/raw/sales_data.csv
git add data/raw/sales_data.csv.dvc data/raw/.gitignore
git commit -m "Track training data with DVC"

# Push data to DVC remote
dvc push
```

### 4. Preprocess Data

```bash
python src/preprocess.py
```

### 5. Train Model

```bash
# Train model with MLflow tracking
python src/train.py

# View experiments in MLflow UI
mlflow ui --port 5001
# Open: http://localhost:5001
```

### 6. Start Services

Open 3 terminal windows:

**Terminal 1: FastAPI Backend**
```bash
cd app
uvicorn backend:app --reload --port 5000
```

**Terminal 2: Streamlit Dashboard**
```bash
streamlit run app/dashboard.py
```

**Terminal 3: MLflow UI**
```bash
mlflow ui --port 5001
```

### 7. Access Applications

- 📊 **Dashboard**: http://localhost:8501
- 🔌 **API Docs**: http://localhost:5000/docs
- 📈 **MLflow UI**: http://localhost:5001

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

1. **Add GitHub Secrets**:
   Go to Repository Settings → Secrets → Actions and add:
   ```
   MLFLOW_TRACKING_URI: https://your-mlflow-server.com
   MLFLOW_USERNAME: your-username
   MLFLOW_PASSWORD: your-password
   DVC_REMOTE_URL: s3://your-bucket/dvc-store
   AWS_ACCESS_KEY_ID: your-aws-key
   AWS_SECRET_ACCESS_KEY: your-aws-secret
   SLACK_WEBHOOK: https://hooks.slack.com/services/YOUR/WEBHOOK
   ```

2. **Trigger Pipeline**:
   ```bash
   # Make a change
   vim params.yaml  # Change max_depth: 10 → 15
   
   # Commit and push
   git add params.yaml
   git commit -m "Tune model: increase max_depth to 15"
   git push origin main
   
   # Pipeline automatically triggers!
   ```

3. **Monitor Pipeline**:
   - Go to GitHub repository → Actions tab
   - Watch pipeline execute in real-time
   - Approve production deployment when ready

### Pipeline Stages

```
1. ✅ Code Quality (2 min)
   ├── Linting (flake8, black)
   └── Unit tests (pytest)

2. ✅ Data Validation (3 min)
   ├── DVC data pull
   ├── Schema validation
   └── Quality checks

3. ✅ Model Training (5-10 min)
   ├── Load data
   ├── Train model
   └── Log to MLflow

4. ✅ Model Evaluation (2 min)
   ├── Compare with baseline
   └── Decision: promote or reject

5. ✅ Model Registration (1 min)
   └── Register in MLflow Registry

6. ✅ Staging Deployment (3 min)
   ├── Deploy to staging
   └── Smoke tests

7. ⏸️  Production Approval (Manual)
   └── Reviewer approves deployment

8. ✅ Production Deployment (5 min)
   ├── Deploy to production
   └── Health checks
```

## 📁 Project Structure

```
ml-forecast-system/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml      # GitHub Actions CI/CD
├── app/
│   ├── backend.py               # FastAPI backend
│   └── dashboard.py             # Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Data preprocessing
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation
│   └── utils.py                 # Helper functions
├── scripts/
│   ├── generate_data.py         # Generate sample data
│   ├── rollback.py              # Rollback procedures
│   └── check_metrics.py         # Monitoring script
├── tests/
│   ├── test_data.py             # Data validation tests
│   ├── test_model.py            # Model tests
│   └── test_api.py              # API tests
├── data/
│   ├── raw/                     # Raw data (DVC tracked)
│   └── processed/               # Processed data
├── models/
│   ├── baseline/                # Baseline models
│   └── trained/                 # Trained models
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Hyperparameters
├── requirements.txt             # Python dependencies
├── .dvcignore                   # DVC ignore patterns
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## 🔧 Configuration

### Hyperparameters (params.yaml)

```yaml
preprocess:
  test_size: 0.2
  random_state: 42

train:
  model_type: RandomForestRegressor
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  random_state: 42
```

### DVC Pipeline (dvc.yaml)

The DVC pipeline defines the complete ML workflow:
1. Preprocess data
2. Train model
3. Evaluate model

Run entire pipeline: `dvc repro`

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Model Performance

Current production model:
- **MAE**: 4.92
- **RMSE**: 7.01
- **R² Score**: 0.89

## 🔄 Data Versioning with DVC

```bash
# Add new data version
dvc add data/raw/sales_data.csv
git commit -m "Update dataset v2"
dvc push

# Switch to previous version
git checkout HEAD~1 data/raw/sales_data.csv.dvc
dvc checkout

# View DVC pipeline
dvc dag
```

## 🏥 Monitoring & Maintenance

### Health Checks

```bash
# Check API health
curl http://localhost:5000/health

# Get metrics
curl http://localhost:5000/metrics
```

### Rollback Procedure

```bash
# Emergency rollback to previous version
python scripts/rollback.py --model sales-forecaster --version 2 --confirm
```

## 📚 Documentation

- **Architecture**: See `docs/architecture.md`
- **API Reference**: http://localhost:5000/docs
- **Model Card**: See `docs/model_card.md`

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and test: `pytest tests/`
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/my-feature`
5. Create Pull Request

## 📝 License

MIT License - See LICENSE file for details

## 🙋 Support

For questions or issues:
- Open an issue on GitHub
- Contact: ml-team@company.com

## 🎓 Learning Resources

This project demonstrates concepts from:
- **MLOps with Agentic AI** (Advanced Certification Course)
- **Session 8**: End-to-End CI/CD for ML with DVC

---

**Built with**: Python • MLflow • DVC • FastAPI • Streamlit • GitHub Actions
