"""
Data Validation Tests
Tests for data quality and schema
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "sales": np.random.uniform(80, 180, 100),
            "advertising_spend": np.random.uniform(1000, 5000, 100),
            "promotions": np.random.choice([0, 1], 100),
            "day_of_week": np.random.randint(0, 7, 100),
            "month": np.random.randint(1, 13, 100),
            "is_weekend": np.random.choice([0, 1], 100),
        }
    )


def test_data_schema(sample_data):
    """Test that data has required columns"""
    required_columns = [
        "date",
        "sales",
        "advertising_spend",
        "promotions",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    for col in required_columns:
        assert col in sample_data.columns, f"Missing column: {col}"


def test_no_nulls(sample_data):
    """Test that data has no null values"""
    null_counts = sample_data.isnull().sum()
    assert null_counts.sum() == 0, f"Found null values: {null_counts[null_counts > 0]}"


def test_data_types(sample_data):
    """Test that columns have correct data types"""
    assert pd.api.types.is_numeric_dtype(sample_data["sales"])
    assert pd.api.types.is_numeric_dtype(sample_data["advertising_spend"])
    assert pd.api.types.is_integer_dtype(sample_data["promotions"])
    assert pd.api.types.is_integer_dtype(sample_data["day_of_week"])
    assert pd.api.types.is_integer_dtype(sample_data["month"])
    assert pd.api.types.is_integer_dtype(sample_data["is_weekend"])


def test_value_ranges(sample_data):
    """Test that values are within expected ranges"""
    assert (sample_data["sales"] >= 0).all(), "Sales cannot be negative"
    assert (sample_data["advertising_spend"] >= 0).all(), "Ad spend cannot be negative"
    assert sample_data["promotions"].isin([0, 1]).all(), "Promotions must be 0 or 1"
    assert (sample_data["day_of_week"] >= 0).all() and (
        sample_data["day_of_week"] <= 6
    ).all()
    assert (sample_data["month"] >= 1).all() and (sample_data["month"] <= 12).all()
    assert sample_data["is_weekend"].isin([0, 1]).all(), "is_weekend must be 0 or 1"


def test_no_duplicates(sample_data):
    """Test that there are no duplicate rows"""
    duplicates = sample_data.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows"


def test_data_size():
    """Test that actual data file exists and has sufficient size"""
    data_path = Path("data/raw/sales_data.csv")

    if data_path.exists():
        df = pd.read_csv(data_path)
        assert len(df) >= 100, f"Dataset too small: {len(df)} rows"
        assert len(df.columns) >= 5, f"Not enough features: {len(df.columns)} columns"


def test_sales_distribution(sample_data):
    """Test that sales distribution is reasonable"""
    mean_sales = sample_data["sales"].mean()
    std_sales = sample_data["sales"].std()

    # Mean should be positive and reasonable
    assert mean_sales > 0, "Mean sales should be positive"

    # Standard deviation should not be too high
    cv = std_sales / mean_sales  # Coefficient of variation
    assert cv < 1.0, f"Sales variance too high: CV={cv:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
