import numpy as np
import pandas as pd
import pytest
from lorepy import uncertainty_plot
from lorepy.uncertainty import _get_feature_importance
from lorepy.lorepy import _prepare_data
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn.svm import SVC


@pytest.fixture
def sample_data():
    X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
    y = [0] * 50 + [1] * 50
    z = X
    return pd.DataFrame({"x": X, "y": y, "z": z})


@pytest.fixture
def custom_colormap():
    return ListedColormap(["red", "green", "blue"])


# Test case for lorepy's uncertainty plot with default parameters
def test_uncertainty_default(sample_data):
    fig, axs = uncertainty_plot(sample_data, "x", "y")  # first test with default params

    assert len(axs) == 2
    assert axs[0].get_title() == "0"
    assert axs[0].get_xlabel() == "x"
    assert axs[0].get_ylabel() == ""


# Test case for lorepy's uncertainty plot with alternative parameters
def test_uncertainty_alternative(sample_data, custom_colormap):
    svc = SVC(probability=True)
    fig, axs = uncertainty_plot(
        sample_data,
        "x",
        "y",
        mode="jackknife",
        x_range=(5, 40),
        colormap=custom_colormap,
        clf=svc,
    )

    assert len(axs) == 2
    assert axs[0].get_title() == "0"
    assert axs[0].get_xlabel() == "x"
    assert axs[0].get_ylabel() == ""


def test_get_uncertainty_confounder(sample_data):
    fig, axs = uncertainty_plot(
        sample_data, "x", "y", confounders=[("z", 5)]
    )  # first test with default params

    assert len(axs) == 2
    assert axs[0].get_title() == "0"
    assert axs[0].get_xlabel() == "x"
    assert axs[0].get_ylabel() == ""


# Test error handling when an unsupported mode is selected
def test_uncertainty_incorrect_mode(sample_data):
    with pytest.raises(NotImplementedError):
        assert uncertainty_plot(sample_data, "x", "y", mode="fail")


def test_uncertainty_with_existing_ax(sample_data):
    fig, ax = plt.subplots(1, 2)  # Create 2 axes manually
    returned_fig, returned_axs = uncertainty_plot(sample_data, "x", "y", ax=ax)

    assert returned_fig is not None
    assert returned_axs[0] == ax[0]
    assert returned_axs[1] == ax[1]
    assert len(returned_axs) == 2
    assert returned_axs[0].get_title() == "0"
    assert returned_axs[0].get_xlabel() == "x"


def test_uncertainty_incorrect_ax_length(sample_data):
    fig, ax = plt.subplots(1, 1)  # Only one axis created, but we expect two
    with pytest.raises(AssertionError):
        uncertainty_plot(sample_data, "x", "y", ax=[ax])


# Test case for feature importance function with default parameters
def test_feature_importance_default(sample_data):
    X_reg, y_reg, _ = _prepare_data(sample_data, "x", "y", [])
    result = _get_feature_importance("x", X_reg, y_reg, iterations=10)
    
    # Check that result is a dictionary with expected keys
    expected_keys = ['feature', 'mean_importance', 'std_importance', 
                    'importance_95ci_low', 'importance_95ci_high',
                    'proportion_positive', 'proportion_negative', 
                    'p_value', 'iterations', 'mode', 'interpretation']
    
    for key in expected_keys:
        assert key in result
    
    # Check basic properties
    assert result['feature'] == "x"
    assert result['iterations'] == 10
    assert result['mode'] == "resample"
    assert isinstance(result['mean_importance'], float)
    assert isinstance(result['p_value'], float)
    assert 0 <= result['p_value'] <= 1
    assert 0 <= result['proportion_positive'] <= 1
    assert 0 <= result['proportion_negative'] <= 1
    # Proportions should sum to <= 1 (the remainder are zeros)
    assert result['proportion_positive'] + result['proportion_negative'] <= 1


# Test case for feature importance with different modes and classifiers
def test_feature_importance_alternative(sample_data):
    X_reg, y_reg, _ = _prepare_data(sample_data, "x", "y", [])
    svc = SVC(probability=True)
    
    result = _get_feature_importance(
        "x", X_reg, y_reg, 
        mode="jackknife",
        iterations=10,
        clf=svc
    )
    
    assert result['mode'] == "jackknife"
    assert result['iterations'] == 10
    assert isinstance(result['mean_importance'], float)


# Test case for feature importance with confounders
def test_feature_importance_confounder(sample_data):
    X_reg, y_reg, _ = _prepare_data(sample_data, "x", "y", [("z", 5)])
    
    result = _get_feature_importance(
        "x", X_reg, y_reg, 
        confounders=[("z", 5)],
        iterations=10
    )
    
    # Should work without errors when confounders are present
    assert result['feature'] == "x"
    assert isinstance(result['mean_importance'], float)


# Test error handling for unsupported mode
def test_feature_importance_incorrect_mode(sample_data):
    X_reg, y_reg, _ = _prepare_data(sample_data, "x", "y", [])
    
    with pytest.raises(NotImplementedError):
        _get_feature_importance("x", X_reg, y_reg, mode="invalid_mode")
