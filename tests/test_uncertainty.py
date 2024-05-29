import pandas as pd
import pytest
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

from lorepy.lorepy import uncertainty_plot

# Generate a sample dataset for testing
X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
y = [0] * 50 + [1] * 50
z = X

df = pd.DataFrame({"x": X, "y": y, "z": z})

colormap = ListedColormap(["red", "green", "blue"])


# Test case for lorepy's uncertainty plot with default parameters
def test_uncertainty_default():
    fig, axs = uncertainty_plot(df, "x", "y")  # first test with default params

    assert len(axs) == 2
    assert axs[0].get_title() == "0"
    assert axs[0].get_xlabel() == "x"
    assert axs[0].get_ylabel() == ""


# Test case for lorepy's uncertainty plot with alternative parameters
def test_uncertainty_alternative():
    svc = SVC(probability=True)

    fig, axs = uncertainty_plot(
        df, "x", "y", mode="jackknife", x_range=(5, 40), colormap=colormap, clf=svc
    )

    assert len(axs) == 2
    assert axs[0].get_title() == "0"
    assert axs[0].get_xlabel() == "x"
    assert axs[0].get_ylabel() == ""


# Test error handling when an unsupported mode is selected
def test_uncertainty_incorrect_mode():
    with pytest.raises(NotImplementedError):
        assert uncertainty_plot(df, "x", "y", mode="fail")
