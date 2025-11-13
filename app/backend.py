"""
FastAPI Backend for Sales Forecaster
Loads model from MLflow Model Registry (DagsHub)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
from typing import List
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Sales Forecaster API",
    description="Production ML API with MLflow Registry",
    version="4.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_version = None
model_metadata = {}


class PredictionRequest(BaseModel):
    advertising_spend: float = Field(..., ge=0, le=10000)
    promotions: int = Field(..., ge=0, le=1)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    confidence: float
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model from MLflow Registry on startup"""
    global model, model_version, model_metadata
    
    logger.info("🚀 Starting Sales Forecaster API...")
    
    # Get MLflow configuration from environment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if mlflow_uri and mlflow_username and mlflow_password:
        logger.info(f"🔗 Connecting to MLflow: {mlflow_uri}")
        
        # Set MLflow credentials
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
        
        try:
            # Load model from Production stage
            model_name = "sales-forecaster"
            model_uri = f"models:/{model_name}/Production"
            
            logger.info(f"📥 Loading model: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model metadata
            client = MlflowClient()
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if prod_versions:
                latest = prod_versions[0]
                model_version = f"v{latest.version}"
                model_metadata = {
                    "version": latest.version,
                    "run_id": latest.run_id,
                    "stage": latest.current_stage,
                    "created_at": latest.creation_timestamp,
                    "source": "DagsHub MLflow Registry"
                }
                logger.info(f"✅ Model loaded: {model_name} {model_version}")
                logger.info(f"   Run ID: {latest.run_id}")
                logger.info(f"   Stage: {latest.current_stage}")
            else:
                logger.warning("⚠️  No Production model found")
                model_version = "Unknown"
        
        except Exception as e:
            logger.error(f"❌ Failed to load from MLflow Registry: {e}")
            logger.info("Trying local fallback...")
            load_local_model()
    else:
        logger.warning("⚠️  MLflow credentials not found in .env")
        logger.info("Loading from local file...")
        load_local_model()
    
    if model:
        logger.info("✅ Sales Forecaster API ready!")
    else:
        logger.error("❌ Model not loaded!")


def load_local_model():
    """Fallback: Load model from local file"""
    global model, model_version, model_metadata
    
    model_path = Path("../models/trained/model.pkl")
    if model_path.exists():
        import joblib
        model = joblib.load(model_path)
        model_version = "Local File"
        model_metadata = {"source": "local_file"}
        logger.info(f"✅ Model loaded from: {model_path}")
    else:
        logger.error("❌ No local model found!")
        model = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Sales Forecaster API",
        "version": "4.0.0",
        "status": "healthy" if model else "model_not_loaded",
        "model_version": model_version,
        "model_source": model_metadata.get("source", "unknown"),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        input_data = pd.DataFrame([{
            'advertising_spend': request.advertising_spend,
            'promotions': request.promotions,
            'day_of_week': request.day_of_week,
            'month': request.month,
            'is_weekend': request.is_weekend
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        confidence = 0.85 if 80 < prediction < 200 else 0.70
        
        logger.info(f"Prediction: {prediction:.2f} (confidence: {confidence:.2f})")
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=model_version,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "sales-forecaster",
        "model_version": model_version,
        "metadata": model_metadata,
        "features": [
            "advertising_spend",
            "promotions",
            "day_of_week",
            "month",
            "is_weekend"
        ],
        "model_type": "RandomForestRegressor",
        "loaded_at": datetime.now().isoformat()
    }


@app.get("/model/reload")
async def reload_model():
    """Reload model from registry (for dashboard refresh)"""
    logger.info("🔄 Reloading model from registry...")
    await load_model()
    
    return {
        "status": "reloaded",
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=5000, reload=True)