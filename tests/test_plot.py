import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lorepy.lorepy import _get_area_df, _get_dots_df, loreplot
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pytest


@pytest.fixture
def sample_data():
    X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
    y = [0] * 50 + [1] * 50
    z = X
    return pd.DataFrame({"x": X, "y": y, "z": z})


@pytest.fixture
def logistic_regression_model():
    X_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    y_reg = np.array([0, 1, 0, 1, 1])
    lg = LogisticRegression()
    lg.fit(X_reg, y_reg)
    return X_reg, y_reg, lg


# Test case for loreplot with default parameters
def test_loreplot_default(sample_data):
    loreplot(sample_data, "x", "y")  # first test without specifying the axis

    fig, ax = plt.subplots()
    loreplot(sample_data, "x", "y", ax=ax)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Test case for loreplot with jitter
def test_loreplot_jitter(sample_data):
    loreplot(sample_data, "x", "y")  # first test without specifying the axis

    fig, ax = plt.subplots()
    loreplot(sample_data, "x", "y", ax=ax, jitter=0.05)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Test case for loreplot with confounder
def test_loreplot_confounder(sample_data):
    loreplot(
        sample_data, "x", "y", confounders=[("z", 1)]
    )  # first test without specifying the axis

    fig, ax = plt.subplots()
    loreplot(sample_data, "x", "y", ax=ax)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Test case for loreplot with custom clf
def test_loreplot_custom_clf(sample_data):
    svc = SVC(probability=True)
    loreplot(sample_data, "x", "y", clf=svc)

    fig, ax = plt.subplots()
    loreplot(sample_data, "x", "y", ax=ax)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Test case for loreplot with custom parameters
def test_loreplot_custom(sample_data):
    fig, ax = plt.subplots()
    loreplot(
        sample_data,
        "x",
        "y",
        add_dots=False,
        x_range=(0, 5),
        ax=ax,
        color=["r", "b"],
        linestyle="-",
    )
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Test case for loreplot with add_dots=True
def test_loreplot_with_dots(sample_data):
    fig, ax = plt.subplots()
    loreplot(sample_data, "x", "y", add_dots=True, ax=ax)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Sample data for testing internal functions
X_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
y_reg = np.array([0, 1, 0, 1, 1])
lg = LogisticRegression()
lg.fit(X_reg, y_reg)


# Test case for _get_dots_df
def test_get_dots_df():
    dots_df = _get_dots_df(X_reg, y_reg, lg, "y")
    assert isinstance(dots_df, DataFrame)
    assert "x" in dots_df.columns
    assert "y" in dots_df.columns
    assert "y_feature" not in dots_df.columns
    assert len(dots_df) == len(X_reg)


# Test case for _get_area_df
def test_get_area_df():
    area_df = _get_area_df(lg, "x", (X_reg.min(), X_reg.max()))
    assert isinstance(area_df, DataFrame)
    assert "x" not in area_df.columns
    assert 0 in area_df.columns
    assert 1 in area_df.columns
    assert len(area_df) == 200
    assert area_df.index[0] == X_reg.min()
    assert area_df.index[-1] == X_reg.max()
