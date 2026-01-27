"""
Shared test fixtures for lorepy tests.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def binary_sample_data(random_seed):
    """
    Create sample data with two classes for binary classification.
    Class 0 tends to have lower x values, class 1 tends to have higher x values.
    """
    X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
    y = [0] * 50 + [1] * 50
    z = X + np.random.randn(100) * 0.5  # Confounder correlated with x
    return pd.DataFrame({"x": X.astype(float), "y": y, "z": z})


@pytest.fixture
def multiclass_sample_data(random_seed):
    """
    Create sample data with three classes for multi-class classification.
    """
    X = np.concatenate(
        [
            np.random.randint(0, 5, 30),
            np.random.randint(3, 8, 30),
            np.random.randint(6, 12, 30),
        ]
    )
    y = [0] * 30 + [1] * 30 + [2] * 30
    z = X + np.random.randn(90) * 0.3
    return pd.DataFrame({"x": X.astype(float), "y": y, "z": z})


@pytest.fixture
def data_with_nan(random_seed):
    """Create sample data with NaN values."""
    X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
    y = [0] * 50 + [1] * 50
    z = X.astype(float)

    # Introduce NaN values
    X = X.astype(float)
    X[5] = np.nan
    X[15] = np.nan
    y[25] = np.nan  # This will become float due to NaN

    df = pd.DataFrame({"x": X, "y": y, "z": z})
    df.loc[25, "y"] = np.nan  # Set after creation to avoid type issues
    return df


@pytest.fixture
def small_deterministic_data():
    """Small, deterministic dataset for precise testing."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "z": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )


@pytest.fixture
def fitted_logistic_model(small_deterministic_data):
    """A fitted logistic regression model on small deterministic data."""
    X = small_deterministic_data[["x"]].values
    y = small_deterministic_data["y"].values
    lg = LogisticRegression()
    lg.fit(X, y)
    return lg, X, y


@pytest.fixture
def fitted_multiclass_model(multiclass_sample_data):
    """A fitted logistic regression model for multi-class classification."""
    X = multiclass_sample_data[["x"]].values
    y = multiclass_sample_data["y"].values
    lg = LogisticRegression(max_iter=200)
    lg.fit(X, y)
    return lg, X, y


@pytest.fixture
def single_class_data():
    """Data with only one class - should cause issues."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0, 0, 0, 0, 0],
            "z": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame({"x": [], "y": [], "z": []})


@pytest.fixture
def string_class_labels(random_seed):
    """Data with string class labels instead of integers."""
    X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
    y = ["class_a"] * 50 + ["class_b"] * 50
    return pd.DataFrame({"x": X.astype(float), "y": y})


@pytest.fixture
def custom_colormap():
    """Custom colormap for testing uncertainty_plot."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(["red", "green", "blue"])
