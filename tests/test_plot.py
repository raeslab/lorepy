import pytest
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from lorepy.lorepy import loreplot, _get_dots_df, _get_area_df

# Generate a sample dataset for testing
X = np.concatenate([np.random.randint(0, 10, 50), np.random.randint(2, 12, 50)])
y = [0] * 50 + [1] * 50

df = pd.DataFrame({"x": X, "y": y})


# Test case for loreplot with default parameters
def test_loreplot_default():
    fig, ax = plt.subplots()
    loreplot(df, "x", "y", ax=ax)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Test case for loreplot with custom parameters
def test_loreplot_custom():
    fig, ax = plt.subplots()
    loreplot(
        df,
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
def test_loreplot_with_dots():
    fig, ax = plt.subplots()
    loreplot(df, "x", "y", add_dots=True, ax=ax)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == ""


# Sample data for testing internal functions
X_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
y_reg = np.array([0, 1, 0, 1, 1])
lg = LogisticRegression(multi_class="multinomial")
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
    area_df = _get_area_df(X_reg, lg, "x")
    assert isinstance(area_df, DataFrame)
    assert "x" not in area_df.columns
    assert 0 in area_df.columns
    assert 1 in area_df.columns
    assert len(area_df) == 200
    assert area_df.index[0] == X_reg.min()
    assert area_df.index[-1] == X_reg.max()


# Test case for _get_area_df with custom x_range
def test_get_area_df_custom_range():
    x_range = (2.0, 4.0)
    area_df = _get_area_df(X_reg, lg, "x", x_range=x_range)
    assert isinstance(area_df, DataFrame)
    assert "x" not in area_df.columns
    assert 0 in area_df.columns
    assert 1 in area_df.columns
    assert len(area_df) == 200
    assert area_df.index[0] == x_range[0]
    assert area_df.index[-1] == x_range[1]
