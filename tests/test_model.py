"""
Model Tests
Tests for model training and predictions
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib


@pytest.fixture
def sample_features():
    """Create sample feature data"""
    return pd.DataFrame(
        {
            "advertising_spend": [3000, 2500, 4000, 1500],
            "promotions": [1, 0, 1, 0],
            "day_of_week": [0, 1, 5, 6],
            "month": [1, 2, 3, 4],
            "is_weekend": [0, 0, 1, 1],
        }
    )


@pytest.fixture
def sample_target():
    """Create sample target data"""
    return pd.Series([120, 110, 150, 140])


@pytest.fixture
def trained_model(sample_features, sample_target):
    """Train a simple model for testing"""
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(sample_features, sample_target)
    return model


def test_model_can_train(sample_features, sample_target):
    """Test that model can be trained"""
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(sample_features, sample_target)

    assert model is not None
    assert hasattr(model, "predict")


def test_model_predictions(trained_model, sample_features):
    """Test that model can make predictions"""
    predictions = trained_model.predict(sample_features)

    assert len(predictions) == len(sample_features)
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    assert all(p > 0 for p in predictions), "All predictions should be positive"


def test_model_performance(trained_model, sample_features, sample_target):
    """Test that model achieves reasonable performance"""
    predictions = trained_model.predict(sample_features)

    # Calculate MAE
    mae = np.mean(np.abs(predictions - sample_target))

    # MAE should be reasonable (not too high)
    assert mae < 50, f"MAE too high: {mae:.2f}"


def test_feature_importance(trained_model):
    """Test that model has feature importances"""
    assert hasattr(trained_model, "feature_importances_")
    importances = trained_model.feature_importances_

    assert len(importances) == 5  # 5 features
    assert all(i >= 0 for i in importances)
    assert np.isclose(sum(importances), 1.0), "Importances should sum to 1"


def test_model_serialization(trained_model, tmp_path):
    """Test that model can be saved and loaded"""
    # Save model
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(trained_model, model_path)

    # Load model
    loaded_model = joblib.load(model_path)

    # Test predictions are the same
    X_test = pd.DataFrame(
        [[3000, 1, 0, 1, 0]],
        columns=[
            "advertising_spend",
            "promotions",
            "day_of_week",
            "month",
            "is_weekend",
        ],
    )

    original_pred = trained_model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)

    assert np.allclose(original_pred, loaded_pred)


def test_prediction_consistency(trained_model, sample_features):
    """Test that predictions are consistent across multiple calls"""
    pred1 = trained_model.predict(sample_features)
    pred2 = trained_model.predict(sample_features)

    assert np.allclose(pred1, pred2), "Predictions should be deterministic"


def test_model_input_validation(trained_model):
    """Test model handles invalid inputs appropriately"""
    # Test with wrong number of features
    with pytest.raises(ValueError):
        wrong_features = pd.DataFrame([[3000, 1, 0]], columns=["a", "b", "c"])
        trained_model.predict(wrong_features)


def test_trained_model_exists():
    """Test that trained model file exists (if training has been run)"""
    model_path = Path("models/trained/model.pkl")

    if model_path.exists():
        # If model exists, test it can be loaded
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, "predict")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
