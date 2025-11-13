"""
Model Evaluation Script
Compares new model against baseline
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load model from file"""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("✓ Model loaded")
    return model


def load_test_data(data_path):
    """Load test data"""
    logger.info(f"Loading test data from {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(["sales"], axis=1, errors="ignore")
    y = df["sales"]

    logger.info(f"✓ Test data loaded: {len(X)} samples")
    return X, y


def evaluate_model(model, X, y):
    """Evaluate model on test data"""
    logger.info("Evaluating model on test set...")

    predictions = model.predict(X)

    metrics = {
        "test_mae": mean_absolute_error(y, predictions),
        "test_rmse": np.sqrt(mean_squared_error(y, predictions)),
        "test_r2": r2_score(y, predictions),
        "test_mape": np.mean(np.abs((y - predictions) / y)) * 100,
    }

    logger.info("Test set performance:")
    logger.info(f"  MAE:  {metrics['test_mae']:.4f}")
    logger.info(f"  RMSE: {metrics['test_rmse']:.4f}")
    logger.info(f"  R²:   {metrics['test_r2']:.4f}")
    logger.info(f"  MAPE: {metrics['test_mape']:.2f}%")

    return metrics, predictions


def get_baseline_metrics():
    """Get baseline model metrics from MLflow"""
    logger.info("Fetching baseline (production) model metrics...")

    try:
        client = MlflowClient()

        # Try to get current production model
        model_version = client.get_model_version_by_alias(
            "sales-forecaster", "Production"
        )
        run = mlflow.get_run(model_version.run_id)

        # Get metrics from the run
        baseline_mae = run.data.metrics.get(
            "mae", run.data.metrics.get("test_mae", 999999)
        )

        logger.info(f"✓ Baseline MAE: {baseline_mae:.4f}")

        return {
            "baseline_mae": baseline_mae,
            "baseline_version": model_version.version,
            "baseline_run_id": model_version.run_id,
        }

    except Exception as e:
        logger.warning(f"Could not fetch baseline model: {e}")
        logger.warning("Assuming this is the first model (no baseline)")

        return {
            # Very high value to ensure new model is better
            "baseline_mae": 999999,
            "baseline_version": "None",
            "baseline_run_id": "None",
        }


def compare_models(new_metrics, baseline_info):
    """Compare new model with baseline"""
    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison")
    logger.info("=" * 60)

    new_mae = new_metrics["test_mae"]
    baseline_mae = baseline_info["baseline_mae"]

    logger.info(f"New Model MAE:      {new_mae:.4f}")
    logger.info(f"Baseline MAE:       {baseline_mae:.4f}")

    if baseline_mae < 999999:  # Valid baseline exists
        improvement = ((baseline_mae - new_mae) / baseline_mae) * 100
        logger.info(f"Improvement:        {improvement:.2f}%")

        if new_mae < baseline_mae:
            decision = "PROMOTE"
            logger.info("Decision:           ✅ PROMOTE TO STAGING")
            logger.info("Reason:             New model outperforms baseline")
        else:
            decision = "REJECT"
            logger.info("Decision:           ❌ REJECT")
            logger.info("Reason:             New model does not improve upon baseline")
    else:
        decision = "PROMOTE"
        improvement = 0.0
        logger.info("Decision:           ✅ PROMOTE TO STAGING")
        logger.info("Reason:             First model (no baseline to compare)")

    logger.info("=" * 60)

    comparison_result = {
        "new_mae": new_mae,
        "baseline_mae": baseline_mae,
        "improvement_percent": improvement,
        "decision": decision,
        "baseline_version": baseline_info["baseline_version"],
    }

    return comparison_result


def save_metrics(metrics, output_path):
    """Save metrics to JSON file"""
    logger.info(f"Saving metrics to {output_path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✓ Metrics saved: {output_path}")


def main():
    """Main evaluation pipeline"""
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument(
        "--model-path", default="models/trained/model.pkl", help="Path to trained model"
    )
    parser.add_argument(
        "--test-data", default="data/processed/test.csv", help="Path to test data"
    )
    parser.add_argument(
        "--output-metrics",
        default="metrics/eval_metrics.json",
        help="Path to save metrics",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 60)

    try:
        # Load model
        model = load_model(args.model_path)

        # Load test data
        X_test, y_test = load_test_data(args.test_data)

        # Evaluate new model
        new_metrics, predictions = evaluate_model(model, X_test, y_test)

        # Get baseline metrics
        baseline_info = get_baseline_metrics()

        # Compare models
        comparison = compare_models(new_metrics, baseline_info)

        # Combine all metrics
        all_metrics = {**new_metrics, **comparison}

        # Save metrics
        save_metrics(all_metrics, args.output_metrics)

        logger.info("\n" + "=" * 60)
        logger.info("✅ Evaluation Complete!")
        logger.info("=" * 60)

        # Exit with appropriate code for CI/CD
        if comparison["decision"] == "PROMOTE":
            logger.info("✓ Model approved for deployment")
            return 0
        else:
            logger.warning("✗ Model rejected, will not deploy")
            return 1

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
