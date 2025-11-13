"""
Model Training Pipeline with MLflow Tracking
Trains Random Forest model and logs to MLflow
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import yaml
import joblib
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_params(config_path="params.yaml"):
    """Load hyperparameters from config"""
    logger.info(f"Loading parameters from {config_path}")
    with open(config_path) as f:
        params = yaml.safe_load(f)
    return params["train"]


def load_data(data_path):
    """Load training data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(["sales"], axis=1, errors="ignore")
    y = df["sales"]

    logger.info(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_model(X, y, params):
    """Train Random Forest model"""
    logger.info("Starting model training...")
    logger.info("Hyperparameters:")
    for key, value in params.items():
        if key != "model_type":
            logger.info(f"  {key}: {value}")

    # Create and train model
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=-1,
        verbose=0,
    )

    start_time = datetime.now()
    model.fit(X, y)
    training_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"✓ Training complete in {training_time:.2f} seconds")

    return model, training_time


def evaluate_model(model, X, y):
    """Evaluate model performance"""
    logger.info("Evaluating model...")

    predictions = model.predict(X)

    metrics = {
        "mae": mean_absolute_error(y, predictions),
        "rmse": np.sqrt(mean_squared_error(y, predictions)),
        "r2_score": r2_score(y, predictions),
        "mape": np.mean(np.abs((y - predictions) / y))
        * 100,  # Mean Absolute Percentage Error
    }

    logger.info("Model performance:")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²:   {metrics['r2_score']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics, predictions


def save_model(model, output_dir):
    """Save model to disk"""
    logger.info(f"Saving model to {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / "model.pkl"
    joblib.dump(model, model_path)

    logger.info(f"✓ Model saved: {model_path}")
    return model_path


def log_to_mlflow(params, metrics, model, training_time, args):
    """Log everything to MLflow"""
    logger.info("Logging to MLflow...")

    # Log parameters
    mlflow.log_params(params)
    mlflow.log_param("training_time_seconds", training_time)

    # Log metrics
    mlflow.log_metrics(metrics)

    # Log tags
    mlflow.set_tag("model_type", params["model_type"])
    mlflow.set_tag("git_commit", args.git_commit)
    mlflow.set_tag("git_author", args.git_author)
    mlflow.set_tag("git_branch", args.git_branch)
    mlflow.set_tag("dvc_data_version", args.dvc_version)
    mlflow.set_tag("environment", "ci-cd" if args.git_commit != "unknown" else "local")
    mlflow.set_tag("training_timestamp", datetime.now().isoformat())

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=None,
        # Don't register yet (will do in CI/CD if good)
        signature=None,
    )

    # Get feature importance
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": [
                    "advertising_spend",
                    "promotions",
                    "day_of_week",
                    "month",
                    "is_weekend",
                ],
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Log as artifact
        importance_path = "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)

        logger.info("\nFeature Importance:")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    logger.info("✓ MLflow logging complete")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train sales forecasting model")
    parser.add_argument(
        "--data-path", default="data/processed/train.csv", help="Path to training data"
    )
    parser.add_argument(
        "--model-output", default="models/trained/", help="Directory to save model"
    )
    parser.add_argument(
        "--experiment-name",
        default="sales-forecaster-dev",
        help="MLflow experiment name",
    )
    parser.add_argument("--git-commit", default="unknown", help="Git commit SHA")
    parser.add_argument("--git-author", default="unknown", help="Git commit author")
    parser.add_argument("--git-branch", default="unknown", help="Git branch name")
    parser.add_argument(
        "--dvc-version", default="unknown", help="DVC data version hash"
    )
    parser.add_argument("--config", default="params.yaml", help="Path to config file")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)

    try:
        # Load parameters
        params = load_params(args.config)

        # Load data
        X, y = load_data(args.data_path)

        # Set MLflow experiment
        mlflow.set_experiment(args.experiment_name)
        logger.info(f"MLflow experiment: {args.experiment_name}")

        # Start MLflow run
        run_name = f"train-{datetime.now():%Y%m%d-%H%M%S}"
        with mlflow.start_run(run_name=run_name):
            logger.info(f"MLflow run: {run_name}")

            # Train model
            model, training_time = train_model(X, y, params)

            # Evaluate model
            metrics, predictions = evaluate_model(model, X, y)

            # Save model locally
            model_path = save_model(model, args.model_output)

            # Log everything to MLflow
            log_to_mlflow(params, metrics, model, training_time, args)

            # Save run ID for CI/CD pipeline
            run_id = mlflow.active_run().info.run_id
            run_id_path = Path(args.model_output) / "run_id.txt"
            with open(run_id_path, "w") as f:
                f.write(run_id)

            logger.info("=" * 60)
            logger.info("✅ Training Complete!")
            logger.info("=" * 60)
            logger.info(f"MLflow Run ID: {run_id}")
            logger.info(f"Model saved: {model_path}")
            logger.info("\nModel Performance Summary:")
            logger.info(f"  MAE:  {metrics['mae']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  R²:   {metrics['r2_score']:.4f}")

            # Quality check
            if metrics["mae"] > 10.0:
                logger.warning("⚠️  MAE is high (>10). Model quality may be poor.")
            else:
                logger.info("✓ Model quality looks good!")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
