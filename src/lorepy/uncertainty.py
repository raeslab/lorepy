import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from lorepy.lorepy import _get_area_df, _prepare_data


def _get_uncertainty_data(
    x: str,
    X_reg,
    y_reg,
    x_range,
    mode="resample",
    jackknife_fraction: float = 0.8,
    iterations: int = 100,
    confounders=None,
    clf=None,
):
    confounders = [] if confounders is None else confounders

    areas = []
    for i in range(iterations):
        if mode == "jackknife":
            X_keep, _, y_keep, _ = train_test_split(
                X_reg, y_reg, train_size=jackknife_fraction
            )
        elif mode == "resample":
            X_keep, y_keep = resample(X_reg, y_reg, replace=True)
        else:
            raise NotImplementedError(
                f"Mode {mode} is unsupported, only jackknife and resample are valid modes"
            )

        lg = LogisticRegression() if clf is None else clf
        lg.fit(X_keep, y_keep)
        new_area = _get_area_df(lg, x, x_range, confounders=confounders).reset_index()

        areas.append(new_area)

    long_df = pd.concat(areas).melt(id_vars=[x]).sort_values(x)

    output = (
        long_df.groupby([x, "variable"])
        .agg(
            min=pd.NamedAgg(column="value", aggfunc="min"),
            mean=pd.NamedAgg(column="value", aggfunc="mean"),
            max=pd.NamedAgg(column="value", aggfunc="max"),
            low_95=pd.NamedAgg(column="value", aggfunc=lambda v: np.percentile(v, 2.5)),
            high_95=pd.NamedAgg(
                column="value", aggfunc=lambda v: np.percentile(v, 97.5)
            ),
            low_50=pd.NamedAgg(column="value", aggfunc=lambda v: np.percentile(v, 25)),
            high_50=pd.NamedAgg(column="value", aggfunc=lambda v: np.percentile(v, 75)),
        )
        .reset_index()
    )

    return output, long_df


def uncertainty_plot(
    data: DataFrame,
    x: str,
    y: str,
    x_range=None,
    mode="resample",
    jackknife_fraction=0.8,
    iterations=100,
    confounders=[],
    colormap=None,
    clf=None,
    ax=None,
):
    """
    Code to create a multi-panel plot, one panel for each category, with the prevalence of that category across the
    range of x-values, along with the uncertainty (intervals containing 50% and 95% of the samples are shown)

    :param data: Pandas dataframe with data
    :param x: Needs to be a numerical feature
    :param y: Categorical feature
    :param x_range: Either None (range will be selected automatically) or a tuple with min and max value for the x-axis
    :param mode: Sampling method, either "resample" (bootstrap) or "jackknife" (default = "resample")
    :param jackknife_fraction: Fraction of data to retain for each jackknife sample (default = 0.8)
    :param iterations: Number of iterations for resampling or jackknife (default = 100)
    :param confounders: List of tuples with the feature and reference value e.g., [("BMI", 25)] will use a reference of 25 for plots
    :param colormap: Colormap to use for the plot, default is None in which case matplotlib's default will be used
    :param clf: Provide a different scikit-learn classifier for the function. Should implement the predict_proba() and fit(). If None a LogisticRegression will be used.
    :param ax: Optional. List of matplotlib Axes to plot into. If None, a new figure and axes will be created.
    :return: A tuple containing the figure and axes objects
    """
    X_reg, y_reg, r = _prepare_data(data, x, y, confounders)

    if x_range is None:
        x_range = r

    plot_df, _ = _get_uncertainty_data(
        x,
        X_reg,
        y_reg,
        x_range,
        mode=mode,
        jackknife_fraction=jackknife_fraction,
        iterations=iterations,
        confounders=confounders,
        clf=clf,
    )

    categories = plot_df.variable.unique()

    if ax is None:
        fig, axs = plt.subplots(ncols=len(categories), sharex=True, sharey=True)
    else:
        assert len(ax) == len(
            categories
        ), "Length of ax must match number of categories"
        fig = ax[0].figure
        axs = ax

    cmap = plt.get_cmap("tab10") if colormap is None else colormap

    for idx, category in enumerate(categories):
        cat_df = plot_df[plot_df.variable == category]

        axs[idx].fill_between(
            cat_df[x], cat_df["low_95"], cat_df["high_95"], alpha=0.1, color=cmap(idx)
        )
        axs[idx].fill_between(
            cat_df[x], cat_df["low_50"], cat_df["high_50"], alpha=0.2, color=cmap(idx)
        )
        axs[idx].plot(cat_df[x], cat_df["mean"], color=cmap(idx))
        axs[idx].set_title(categories[idx])
        axs[idx].set_xlabel(x)

        axs[idx].set_xlim(*x_range)
        axs[idx].set_ylim(0, 1)

    return fig, axs
