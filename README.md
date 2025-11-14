# 🚀 Sales Forecaster - Production MLOps Pipeline

**Complete End-to-End Automated ML System with CI/CD**

[![MLOps](https://img.shields.io/badge/MLOps-Production-green)](https://github.com/yourusername/ml-forecast-system)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.0-orange)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.27.0-blueviolet)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-teal)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io/)

---

## 📋 What is This?

This is a **production-grade MLOps pipeline** that demonstrates industry best practices for:

- ✅ **Automated ML Training** - Train models with one git push
- ✅ **Model Versioning** - MLflow Model Registry integration
- ✅ **Data Versioning** - DVC for reproducible datasets
- ✅ **CI/CD Automation** - GitHub Actions pipeline
- ✅ **Real-time Serving** - FastAPI backend with auto-reload
- ✅ **Interactive Dashboard** - Streamlit UI with live metrics
- ✅ **Experiment Tracking** - DagsHub integration

**Perfect for:**
- MLOps courses and training
- Portfolio projects
- Learning production ML workflows
- Interview preparation

---

## 🎯 Learning Path

### **Step 1: Learn Concepts (30 minutes)**
📓 **Start here:** [`Session_8_MLOps_Pipeline_Foundation.ipynb`](Session_8_MLOps_Pipeline_Foundation.ipynb)

This Jupyter notebook teaches you:
- Why MLOps matters (real production disasters!)
- Core concepts: CI/CD, Model Registry, Data Versioning
- Hands-on exercises with explanations
- Smooth transition to the full implementation

**Run the notebook first to understand WHAT and WHY before building.**

---

### **Step 2: Build the System (60-90 minutes)**
📄 **Implementation guide:** [`MLOps_Complete_Setup_Guide.md`](MLOps_Complete_Setup_Guide.md)

Complete step-by-step instructions for:
- Windows & macOS
- Anaconda Prompt or VS Code
- GitHub, DagsHub, and local setup
- Troubleshooting common issues

**Follow this guide to build your production pipeline.**

---

## ⚡ Quick Start (For Experienced Users)

If you're already familiar with MLOps concepts:

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/ml-forecast-system.git
cd ml-forecast-system

# 2. Create environment
conda create -n mlops python=3.10 -y
conda activate mlops
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env with your DagsHub credentials

# 4. Initialize DVC
dvc init
dvc remote add origin https://dagshub.com/YOUR_USERNAME/ml-forecast-system.dvc

# 5. Generate data and train
python scripts/generate_data.py --n-days 1065
python src/preprocess.py
python src/train.py

# 6. Start services
# Terminal 1:
uvicorn app.backend:app --reload --port 5000

# Terminal 2:
streamlit run app/dashboard.py

# 7. Make changes and watch automation!
# Edit params.yaml, commit, push → Full CI/CD runs automatically
```

---

## 📁 Project Structure

```
ml-forecast-system/
│
├── 📓 Session_8_MLOps_Pipeline_Foundation.ipynb   # Learning notebook
├── 📄 MLOps_Complete_Setup_Guide.md               # Implementation guide
│
├── .github/                    # CI/CD Configuration
│   ├── scripts/
│   │   ├── register_model.py           # MLflow model registration
│   │   └── validate_data.py            # Data quality checks
│   └── workflows/
│       └── ml-pipeline.yml             # GitHub Actions workflow
│
├── app/                        # Production Applications
│   ├── backend.py                      # FastAPI REST API (auto-reload)
│   └── dashboard.py                    # Streamlit dashboard (real-time metrics)
│
├── data/                       # Data Storage (DVC tracked)
│   ├── raw/                            # Original datasets
│   └── processed/                      # Train/test splits
│
├── models/                     # Model Artifacts
│   └── trained/                        # Trained models
│
├── src/                        # ML Pipeline Code
│   ├── preprocess.py                   # Data preprocessing
│   ├── train.py                        # Model training (MLflow)
│   ├── evaluate.py                     # Model evaluation
│   └── utils.py                        # Helper functions
│
├── scripts/                    # Utility Scripts
│   └── generate_data.py                # Synthetic data generation
│
├── tests/                      # Unit Tests
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_utils.py
│
├── .env.example                # Template for credentials
├── .gitignore                  # Git ignore rules
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Model hyperparameters
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Tech Stack

### **ML & Data**
- **Scikit-learn** - Model training (RandomForest)
- **Pandas & NumPy** - Data manipulation
- **MLflow** - Experiment tracking & model registry
- **DVC** - Data version control

### **Backend & API**
- **FastAPI** - REST API with auto-reload
- **Uvicorn** - ASGI server
- **Python-dotenv** - Environment management

### **Frontend & Visualization**
- **Streamlit** - Interactive dashboard
- **Plotly** - Charts and visualizations

### **DevOps & CI/CD**
- **GitHub Actions** - CI/CD pipeline
- **DagsHub** - MLflow & DVC hosting
- **pytest** - Unit testing
- **flake8 & black** - Code quality

---

## 🎓 What You'll Learn

### **MLOps Fundamentals**
- Experiment tracking and reproducibility
- Model versioning and registry
- Data versioning with DVC
- CI/CD for machine learning

### **Production Patterns**
- Automated model training pipelines
- Model serving with REST APIs
- Real-time dashboards
- Auto-reload and hot-swapping models

### **DevOps Practices**
- Infrastructure as Code
- Automated testing
- GitHub Actions workflows
- Secrets management

### **Tools Mastery**
- MLflow for experiment tracking
- DVC for data versioning
- FastAPI for model serving
- Streamlit for dashboards

---

## 🚀 How It Works

### **The Complete Automation Flow:**

```
1. Developer changes code (e.g., hyperparameters)
   ↓
2. Git push to GitHub
   ↓
3. GitHub Actions automatically triggers
   ↓
4. Pipeline runs (6 automated jobs):
   - Code quality checks
   - Data validation
   - Model training
   - Model evaluation
   - Model registration in MLflow
   - Deployment complete
   ↓
5. Model registered in DagsHub MLflow Registry
   ↓
6. Backend detects new model (30s polling)
   ↓
7. Backend auto-reloads new model
   ↓
8. Dashboard auto-refreshes (30s polling)
   ↓
9. New predictions served automatically!
```

**Total time:** ~15-20 minutes from push to production

**Manual steps:** ZERO! ✨

---

## 📊 Live Applications

Once deployed, you'll have:

### **1. REST API (FastAPI)**
- **URL:** http://localhost:5000
- **Docs:** http://localhost:5000/docs
- **Features:**
  - `/predict` - Make predictions
  - `/model/info` - Get model metadata
  - `/model/compare` - Compare versions
  - Auto-reload when new models deploy

### **2. Dashboard (Streamlit)**
- **URL:** http://localhost:8501
- **Features:**
  - Real-time model metrics
  - 7-day sales forecasts
  - Custom predictions
  - Model version tracking
  - Auto-refresh every 30s

---

## 🎯 Use Cases

This system demonstrates:

1. **Sales Forecasting** (Current Implementation)
   - Predict daily sales based on advertising, promotions, seasonality
   - Track model performance over time
   - A/B test different models

2. **Easily Adaptable To:**
   - Demand forecasting
   - Customer churn prediction
   - Price optimization
   - Fraud detection
   - Any supervised learning problem

---

## 🔧 Configuration

### **Environment Variables (.env)**

```bash
# DagsHub credentials
DAGSHUB_USERNAME=your-username
DAGSHUB_TOKEN=ghp_your_token
DAGSHUB_REPO=ml-forecast-system

# MLflow (auto-configured from DagsHub)
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/ml-forecast-system.mlflow
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=ghp_your_token

# Auto-reload settings
AUTO_RELOAD_MODEL=true
AUTO_RELOAD_INTERVAL=30
```

### **Model Hyperparameters (params.yaml)**

```yaml
train:
  model_type: RandomForestRegressor
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  random_state: 42

preprocess:
  test_size: 0.2
  random_state: 42
```

---

## 🧪 Running Tests

```bash
# Activate environment
conda activate mlops

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term

# Run specific test file
pytest tests/test_train.py -v

# Run code quality checks
flake8 src/ tests/ --max-line-length=100
black --check src/ tests/
```

---

## 📈 Experiment Tracking

All experiments are tracked in DagsHub:

**View Experiments:**
https://dagshub.com/YOUR_USERNAME/ml-forecast-system/experiments

**What's Tracked:**
- Model hyperparameters
- Training metrics (MAE, RMSE, R²)
- Git commit hash
- Training duration
- Dataset version (DVC hash)

**Model Registry:**
https://dagshub.com/YOUR_USERNAME/ml-forecast-system.mlflow/#/models

**What's Stored:**
- Model versions
- Stage (Production/Staging/Archived)
- Performance metrics
- Source run ID

---

## 🎨 Customization

### **Change the Model**

Edit `src/train.py`:
```python
# Current: RandomForest
from sklearn.ensemble import RandomForestRegressor

# Try: XGBoost
from xgboost import XGBRegressor

model = XGBRegressor(**params)
```

### **Add Features**

Edit `src/preprocess.py`:
```python
# Add moving average
df['sales_ma_7'] = df['sales'].rolling(7).mean()

# Add holiday indicator
df['is_holiday'] = df['date'].isin(holidays)
```

### **Change CI/CD Behavior**

Edit `.github/workflows/ml-pipeline.yml`:
```yaml
# Add deployment approval
- name: Wait for approval
  uses: trstringer/manual-approval@v1
```

---

## 🐛 Troubleshooting

### **Common Issues:**

1. **Backend can't connect to MLflow**
   - Check `.env` file exists and has correct credentials
   - Verify `MLFLOW_TRACKING_URI` format
   - Restart backend

2. **Dashboard shows "Local File"**
   - Wait 30s for auto-reload
   - Manually reload: http://localhost:5000/model/reload
   - Check model registered in DagsHub

3. **CI/CD fails**
   - Check GitHub Secrets are configured
   - Verify DagsHub token is valid
   - Review workflow logs

**Full troubleshooting guide:** See [`MLOps_Complete_Setup_Guide.md`](MLOps_Complete_Setup_Guide.md#troubleshooting)

---

## 📚 Additional Resources

### **Documentation**
- **MLflow:** https://mlflow.org/docs/latest/
- **DVC:** https://dvc.org/doc
- **FastAPI:** https://fastapi.tiangolo.com/
- **Streamlit:** https://docs.streamlit.io/

### **Learning Resources**
- **MLOps Community:** https://mlops.community/
- **Made With ML:** https://madewithml.com/
- **Full Stack Deep Learning:** https://fullstackdeeplearning.com/

### **Similar Projects**
- **MLOps Zoomcamp:** https://github.com/DataTalksClub/mlops-zoomcamp
- **Awesome MLOps:** https://github.com/visenger/awesome-mlops

---

## 🤝 Contributing

This is a course project template. Feel free to:
- Fork and customize for your needs
- Add new features
- Improve documentation
- Share with others

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

**Course:** MLOps with Agentic AI - Advanced Certification Course

**Tools:**
- Anthropic MLflow for experiment tracking
- DagsHub for MLflow & DVC hosting
- GitHub Actions for CI/CD
- FastAPI & Streamlit for applications

---

## 📞 Support

**Having issues?**
1. Check the [Setup Guide](MLOps_Complete_Setup_Guide.md)
2. Review the [Troubleshooting Section](MLOps_Complete_Setup_Guide.md#troubleshooting)
3. Run the [Concepts Notebook](Session_8_MLOps_Pipeline_Foundation.ipynb)
4. Contact course instructors

---

## 🎯 Next Steps

**After completing this project:**

1. **Enhance the Pipeline**
   - Add model monitoring (Prometheus + Grafana)
   - Implement A/B testing
   - Add data drift detection

2. **Scale Up**
   - Deploy to Kubernetes
   - Use cloud services (AWS SageMaker, GCP Vertex AI)
   - Add distributed training

3. **Apply to Real Projects**
   - Use your own dataset
   - Solve real business problems
   - Build your portfolio

4. **Learn Advanced Topics**
   - LLMOps (Module 4 of course)
   - Agentic AI systems
   - Multi-model serving

---

**⭐ Star this repo if you found it helpful!**

**📢 Share your implementation on LinkedIn with #MLOps**

**🚀 Keep learning, keep building!**

---

**Version:** 1.0  
**Last Updated:** November 2024  
**Course:** MLOps with Agentic AI
