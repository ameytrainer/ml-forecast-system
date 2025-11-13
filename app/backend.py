"""
FastAPI Backend for Sales Forecaster
Production ML API with MLflow Model Registry Integration
Features: Auto-reload, Real Metrics, Model Comparison
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
from threading import Thread
import time

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Sales Forecaster API",
    description="Production ML API with MLflow Registry & Auto-Reload",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
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

# Auto-reload configuration
AUTO_RELOAD_ENABLED = os.getenv("AUTO_RELOAD_MODEL", "true").lower() == "true"
AUTO_RELOAD_INTERVAL = int(os.getenv("AUTO_RELOAD_INTERVAL", "30"))


class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    advertising_spend: float = Field(..., ge=0, le=10000, description="Advertising spend in dollars")
    promotions: int = Field(..., ge=0, le=1, description="Whether promotions are active (0 or 1)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="Whether it's weekend (0 or 1)")


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: float
    model_version: str
    confidence: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


def check_for_new_model():
    """
    Background task to check for new model versions in MLflow Registry
    Runs in a separate thread and auto-reloads model when new version is detected
    """
    global model, model_version, model_metadata
    
    logger.info(f"🔄 Auto-reload enabled: checking every {AUTO_RELOAD_INTERVAL}s")
    
    while AUTO_RELOAD_ENABLED:
        time.sleep(AUTO_RELOAD_INTERVAL)
        
        try:
            # Skip if not using MLflow Registry
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not mlflow_uri or not mlflow_uri.startswith("http"):
                continue
            
            # Skip if model not loaded yet
            if not model_metadata.get("version"):
                continue
            
            client = MlflowClient()
            model_name = "sales-forecaster"
            
            # Get current Production version from registry
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if not prod_versions:
                continue
            
            latest_version = prod_versions[0]
            current_version = model_metadata.get("version")
            
            # Check if version changed
            if str(latest_version.version) != str(current_version):
                logger.info(f"🆕 New model version detected!")
                logger.info(f"   Current: v{current_version}")
                logger.info(f"   Latest: v{latest_version.version}")
                logger.info(f"🔄 Auto-reloading model...")
                
                # Reload model
                try:
                    model_uri = f"models:/{model_name}/Production"
                    new_model = mlflow.pyfunc.load_model(model_uri)
                    
                    # Update globals atomically
                    model = new_model
                    model_version = f"v{latest_version.version}"
                    model_metadata = {
                        "version": latest_version.version,
                        "run_id": latest_version.run_id,
                        "stage": latest_version.current_stage,
                        "created_at": latest_version.creation_timestamp,
                        "source": "DagsHub MLflow Registry"
                    }
                    
                    logger.info(f"✅ Model auto-reloaded to version {model_version}!")
                    logger.info(f"   Run ID: {latest_version.run_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to reload model: {e}")
                    
        except Exception as e:
            logger.error(f"Error in auto-reload check: {e}")


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
    
    # Start auto-reload thread if model loaded successfully and auto-reload enabled
    if model and AUTO_RELOAD_ENABLED and mlflow_uri:
        reload_thread = Thread(target=check_for_new_model, daemon=True)
        reload_thread.start()
        logger.info(f"✅ Auto-reload thread started (checking every {AUTO_RELOAD_INTERVAL}s)")
    
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
    """Root endpoint with API information"""
    return {
        "service": "Sales Forecaster API",
        "version": "4.0.0",
        "status": "healthy" if model else "model_not_loaded",
        "model_version": model_version,
        "model_source": model_metadata.get("source", "unknown"),
        "auto_reload_enabled": AUTO_RELOAD_ENABLED,
        "auto_reload_interval": f"{AUTO_RELOAD_INTERVAL}s" if AUTO_RELOAD_ENABLED else "disabled",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_debug": "/predict/debug",
            "model_info": "/model/info",
            "model_compare": "/model/compare",
            "reload": "/model/reload",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make sales prediction
    
    Returns predicted sales value based on input features.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact support."
        )
    
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            'advertising_spend': request.advertising_spend,
            'promotions': request.promotions,
            'day_of_week': request.day_of_week,
            'month': request.month,
            'is_weekend': request.is_weekend
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence (simplified for demo)
        confidence = 0.85 if 80 < prediction < 200 else 0.70
        
        logger.info(f"Prediction: {prediction:.2f} (confidence: {confidence:.2f}) [Model: {model_version}]")
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=model_version,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/debug", tags=["Predictions"])
async def predict_debug(request: PredictionRequest):
    """
    Make prediction with detailed debug information
    Shows which model is being used and full metadata
    """
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
        
        logger.info(f"DEBUG Prediction: {prediction:.2f} using {model_version}")
        
        return {
            "prediction": float(prediction),
            "model_version": model_version,
            "model_source": model_metadata.get("source", "unknown"),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "input_features": {
                "advertising_spend": request.advertising_spend,
                "promotions": request.promotions,
                "day_of_week": request.day_of_week,
                "month": request.month,
                "is_weekend": request.is_weekend
            },
            "model_metadata": {
                "version": model_metadata.get("version", "unknown"),
                "run_id": model_metadata.get("run_id", "unknown"),
                "stage": model_metadata.get("stage", "unknown"),
                "loaded_from": "DagsHub MLflow Registry" if "run_id" in model_metadata else "Local File"
            },
            "auto_reload": {
                "enabled": AUTO_RELOAD_ENABLED,
                "interval_seconds": AUTO_RELOAD_INTERVAL
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """
    Get comprehensive model information with performance metrics
    Fetches real metrics from MLflow run
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Try to fetch metrics from MLflow
    performance_metrics = {}
    
    if model_metadata.get("run_id"):
        try:
            import mlflow
            
            run = mlflow.get_run(model_metadata["run_id"])
            
            # Extract metrics from the run
            metrics = run.data.metrics
            performance_metrics = {
                "mae": metrics.get("mae", metrics.get("test_mae", 0)),
                "rmse": metrics.get("rmse", metrics.get("test_rmse", 0)),
                "r2_score": metrics.get("r2_score", metrics.get("test_r2", 0)),
                "mape": metrics.get("mape", metrics.get("test_mape", 0))
            }
            
            logger.info(f"✓ Fetched metrics from MLflow run: {model_metadata['run_id']}")
            
        except Exception as e:
            logger.warning(f"Could not fetch metrics from MLflow: {e}")
            performance_metrics = {
                "mae": 0,
                "rmse": 0,
                "r2_score": 0,
                "mape": 0
            }
    else:
        logger.warning("No run_id in metadata, using placeholder metrics")
        performance_metrics = {
            "mae": 0,
            "rmse": 0,
            "r2_score": 0,
            "mape": 0
        }
    
    # Add performance to metadata
    enhanced_metadata = model_metadata.copy()
    enhanced_metadata["performance"] = performance_metrics
    
    return {
        "model_name": "sales-forecaster",
        "model_version": model_version,
        "metadata": enhanced_metadata,
        "features": [
            "advertising_spend",
            "promotions",
            "day_of_week",
            "month",
            "is_weekend"
        ],
        "target": "sales",
        "model_type": "RandomForestRegressor",
        "loaded_at": datetime.now().isoformat(),
        "performance": performance_metrics,
        "auto_reload": {
            "enabled": AUTO_RELOAD_ENABLED,
            "interval_seconds": AUTO_RELOAD_INTERVAL,
            "last_check": "monitoring..."
        }
    }


@app.get("/model/compare")
async def compare_models():
    """
    Compare current Production model with previous version
    Returns performance deltas for metrics display
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        client = MlflowClient()
        model_name = "sales-forecaster"
        
        # Get all versions
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        if len(all_versions) < 1:
            return {
                "error": "No model versions found",
                "has_comparison": False
            }
        
        # Sort by version number (descending)
        sorted_versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)
        
        # Get current production version
        prod_versions = [v for v in sorted_versions if v.current_stage == "Production"]
        
        if not prod_versions:
            return {
                "error": "No Production model found",
                "has_comparison": False
            }
        
        current_version = prod_versions[0]
        current_run = mlflow.get_run(current_version.run_id)
        current_metrics = current_run.data.metrics
        
        # Try to find previous version (archived or older)
        previous_version = None
        for v in sorted_versions:
            if int(v.version) < int(current_version.version):
                previous_version = v
                break
        
        if previous_version:
            previous_run = mlflow.get_run(previous_version.run_id)
            previous_metrics = previous_run.data.metrics
            
            # Calculate deltas
            current_mae = current_metrics.get("mae", current_metrics.get("test_mae", 0))
            previous_mae = previous_metrics.get("mae", previous_metrics.get("test_mae", 0))
            
            current_rmse = current_metrics.get("rmse", current_metrics.get("test_rmse", 0))
            previous_rmse = previous_metrics.get("rmse", previous_metrics.get("test_rmse", 0))
            
            current_r2 = current_metrics.get("r2_score", current_metrics.get("test_r2", 0))
            previous_r2 = previous_metrics.get("r2_score", previous_metrics.get("test_r2", 0))
            
            # Calculate percentage changes
            mae_delta = ((previous_mae - current_mae) / previous_mae * 100) if previous_mae > 0 else 0
            rmse_delta = ((previous_rmse - current_rmse) / previous_rmse * 100) if previous_rmse > 0 else 0
            r2_delta = ((current_r2 - previous_r2) / previous_r2 * 100) if previous_r2 > 0 else 0
            
            return {
                "has_comparison": True,
                "current_version": {
                    "version": current_version.version,
                    "run_id": current_version.run_id,
                    "metrics": {
                        "mae": current_mae,
                        "rmse": current_rmse,
                        "r2_score": current_r2
                    }
                },
                "previous_version": {
                    "version": previous_version.version,
                    "run_id": previous_version.run_id,
                    "metrics": {
                        "mae": previous_mae,
                        "rmse": previous_rmse,
                        "r2_score": previous_r2
                    }
                },
                "deltas": {
                    "mae_percent": round(mae_delta, 2),
                    "rmse_percent": round(rmse_delta, 2),
                    "r2_percent": round(r2_delta, 2)
                },
                "improvement": {
                    "mae": "improved" if mae_delta > 0 else "degraded" if mae_delta < 0 else "unchanged",
                    "rmse": "improved" if rmse_delta > 0 else "degraded" if rmse_delta < 0 else "unchanged",
                    "r2": "improved" if r2_delta > 0 else "degraded" if r2_delta < 0 else "unchanged"
                }
            }
        else:
            # No previous version to compare
            current_mae = current_metrics.get("mae", current_metrics.get("test_mae", 0))
            current_rmse = current_metrics.get("rmse", current_metrics.get("test_rmse", 0))
            current_r2 = current_metrics.get("r2_score", current_metrics.get("test_r2", 0))
            
            return {
                "has_comparison": False,
                "message": "This is the first model version",
                "current_version": {
                    "version": current_version.version,
                    "run_id": current_version.run_id,
                    "metrics": {
                        "mae": current_mae,
                        "rmse": current_rmse,
                        "r2_score": current_r2
                    }
                }
            }
            
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return {
            "error": str(e),
            "has_comparison": False
        }


@app.get("/model/reload")
async def reload_model():
    """Manually reload model from registry"""
    logger.info("🔄 Manual reload requested...")
    await load_model()
    
    return {
        "status": "reloaded",
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
        "auto_reload_enabled": AUTO_RELOAD_ENABLED
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=5000, reload=True)