"""
Data Preprocessing Pipeline
Prepares raw data for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path="params.yaml"):
    """Load preprocessing configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["preprocess"]


def load_raw_data(data_path):
    """Load raw data from CSV"""
    logger.info(f"Loading raw data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_data(df):
    """Validate data quality"""
    logger.info("Validating data quality...")

    # Check for required columns
    required_cols = [
        "date",
        "sales",
        "advertising_spend",
        "promotions",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
        # Handle nulls (for now, raise error)
        raise ValueError("Dataset contains null values")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows")
        df = df.drop_duplicates()
        logger.info(f"✓ Removed {duplicates} duplicate rows")

    # Check data types
    if df["sales"].dtype not in [np.float64, np.int64]:
        logger.warning("Converting sales to numeric")
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    logger.info("✓ Data validation complete")
    return df


def engineer_features(df):
    """Create additional features"""
    logger.info("Engineering features...")

    # Convert date to datetime if not already
    if df["date"].dtype == "object":
        df["date"] = pd.to_datetime(df["date"])

    # Add more features (optional)
    # Example: day of month, quarter, etc.
    # df['day_of_month'] = df['date'].dt.day
    # df['quarter'] = df['date'].dt.quarter

    logger.info("✓ Feature engineering complete")
    return df


def prepare_features(df):
    """Prepare features for modeling"""
    logger.info("Preparing features...")

    # Select feature columns (exclude date and target)
    feature_cols = [
        "advertising_spend",
        "promotions",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    # Create feature matrix X and target vector y
    X = df[feature_cols].copy()
    y = df["sales"].copy()

    # Keep date for reference (but don't use in model)
    dates = df["date"].copy() if "date" in df.columns else None

    logger.info(f"✓ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")

    return X, y, dates


def split_data(X, y, dates, test_size, random_state):
    """Split data into train and test sets"""
    logger.info(f"Splitting data (test_size={test_size})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if dates is not None:
        dates_train, dates_test = train_test_split(
            dates, test_size=test_size, random_state=random_state
        )
    else:
        dates_train, dates_test = None, None

    logger.info(f"✓ Train set: {X_train.shape[0]} samples")
    logger.info(f"✓ Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """Save processed data to disk"""
    logger.info(f"Saving processed data to {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine features and target
    train_df = X_train.copy()
    train_df["sales"] = y_train.values

    test_df = X_test.copy()
    test_df["sales"] = y_test.values

    # Save to CSV
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"✓ Train data saved: {train_path}")
    logger.info(f"✓ Test data saved: {test_path}")


def main():
    """Main preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("=" * 60)

    try:
        # Load configuration
        config = load_config()

        # Load raw data
        raw_data_path = "data/raw/sales_data.csv"
        df = load_raw_data(raw_data_path)

        # Validate data
        df = validate_data(df)

        # Engineer features
        df = engineer_features(df)

        # Prepare features
        X, y, dates = prepare_features(df)

        # Split data
        X_train, X_test, y_train, y_test, dates_train, dates_test = split_data(
            X,
            y,
            dates,
            test_size=config["test_size"],
            random_state=config["random_state"],
        )

        # Save processed data
        save_processed_data(X_train, X_test, y_train, y_test, "data/processed")

        logger.info("=" * 60)
        logger.info("✅ Data Preprocessing Complete!")
        logger.info("=" * 60)

        # Summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info("  Target variable: sales")
        logger.info(f"  Sales range: ${y.min():.2f} - ${y.max():.2f}")
        logger.info(f"  Sales mean: ${y.mean():.2f}")

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
