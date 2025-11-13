"""
Register Model in MLflow Model Registry
Run during GitHub Actions CI/CD pipeline
"""

import sys
import os
import mlflow
from mlflow.tracking import MlflowClient


def register_model(run_id: str, model_name: str = "sales-forecaster"):
    """
    Register model from MLflow run into Model Registry
    
    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
    
    Returns:
        model_version: Registered model version number
    """
    try:
        print(f"📦 Registering model from run: {run_id}")
        print(f"   Model name: {model_name}")
        print(f"   MLflow URI: {os.getenv('MLFLOW_TRACKING_URI')}")
        print()
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        print(f"📝 Registering model from URI: {model_uri}")
        
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Model registered successfully!")
        print(f"   Model: {model_name}")
        print(f"   Version: {model_version.version}")
        print()
        
        # Transition to Production stage
        print(f"🚀 Transitioning to Production stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Model promoted to Production!")
        print(f"   Previous versions archived")
        print()
        
        # Add description
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Trained via GitHub Actions CI/CD. Run ID: {run_id}"
        )
        
        # Print summary
        print("=" * 60)
        print("✅ MODEL REGISTRATION COMPLETE")
        print("=" * 60)
        print(f"Model Name: {model_name}")
        print(f"Version: {model_version.version}")
        print(f"Stage: Production")
        print(f"Run ID: {run_id}")
        print("=" * 60)
        
        return model_version.version
        
    except Exception as e:
        print(f"❌ Error registering model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Get arguments
    if len(sys.argv) < 2:
        print("Usage: python register_model.py <run_id> [model_name]")
        sys.exit(1)
    
    run_id = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "sales-forecaster"
    
    # Register model
    version = register_model(run_id, model_name)
    
    # Write version to file for GitHub Actions output
    with open("model_version.txt", "w") as f:
        f.write(str(version))
    
    print(f"\n📄 Model version written to: model_version.txt")