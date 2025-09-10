import numpy as np
import pandas as pd
import pytest
from lorepy import uncertainty_plot
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
