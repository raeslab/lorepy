from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression


def _prepare_data(data, x, y, confounders):
    x_features = [x] + [i[0] for i in confounders]

    tmp_df = data[x_features + [y]].dropna()
    X_reg = np.array(tmp_df[x_features])
    y_reg = np.array(tmp_df[y])

    x_range = (X_reg[:, 0].min(), X_reg[:, 0].max())

    return X_reg, y_reg, x_range


def _get_area_df(lg, x_feature, x_range, confounders=[]) -> DataFrame:
    values = np.linspace(x_range[0], x_range[1], num=200)

    predict_df = pd.DataFrame({"values": values})

    for k, v in confounders:
        predict_df[k] = v

    proba = lg.predict_proba(predict_df.values)
    proba_df = DataFrame(proba, columns=lg.classes_)
    proba_df[x_feature] = values
    proba_df.set_index(x_feature, inplace=True)

    return proba_df


def _get_dots_df(X, y, lg, y_feature, confounders=[], jitter=0) -> DataFrame:
    output = []

    for x, s in zip(X, y):
        if jitter != 0:
            x[0] += np.random.uniform(low=-jitter, high=jitter)

        proba = lg.predict_proba([x] + [i[1] for i in confounders])
        i = list(lg.classes_).index(s)
        min_value = sum(proba[0][:i])
        max_value = sum(proba[0][: i + 1])
        margin = (max_value - min_value) / 10
        ypos = np.random.uniform(low=min_value + margin, high=max_value - margin)
        output.append({y_feature: s, "x": x[0], "y": ypos})

    return DataFrame(output)


def loreplot(
    data: DataFrame,
    x: str,
    y: str,
    add_dots: bool = True,
    x_range: Optional[Tuple[float, float]] = None,
    scatter_kws: dict = dict({}),
    ax=None,
    clf=None,
    confounders=[],
    jitter=0,
    **kwargs,
):
    """
    Code to create a loreplot with a numerical feature on the v-axis and categorical y from a pandas dataset

    :param data: Pandas dataframe with data
    :param x: Needs to be a numerical feature
    :param y: Categorical feature
    :param add_dots: Shows where true samples are in the plot (cannot be enabled when deconfounding for additional variables)
    :param x_range: Either None (range will be selected automatically) or a tuple with min and max value for the v-axis
    :param scatter_kws: Dictionary with keyword arguments to pass to the scatter function
    :param ax: subplot to draw on, in case lorepy is used in a subplot
    :param clf: provide a different scikit-learn classifier for the function. Should implement the predict_proba() and fit()
    :param confounders: list of tuples with the feature and reference value e.g. [("BMI", 25)] will confounders BMI and use a reference of 25 for plots
    :param jitter: adds random noise to the x-position of dots. This can help avoid overplotting when integer values are used for the numerical features
    :param kwargs: Additional arguments to pass to pandas' plot.area function
    """
    if ax is None:
        ax = plt.gca()

    X_reg, y_reg, r = _prepare_data(data, x, y, confounders)

    if x_range is None:
        x_range = r

    lg = LogisticRegression() if clf is None else clf
    lg.fit(X_reg, y_reg)

    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = "None"

    area_df = _get_area_df(lg, x, x_range, confounders=confounders)
    area_df.plot.area(ax=ax, **kwargs)

    if add_dots and len(confounders) == 0:
        dot_df = _get_dots_df(X_reg, y_reg, lg, y, jitter=jitter)
        if "color" not in scatter_kws.keys():
            scatter_kws["color"] = "w"
        if "alpha" not in scatter_kws.keys():
            scatter_kws["alpha"] = 0.3
        ax.scatter(dot_df["x"], dot_df["y"], **scatter_kws)

    ax.set_xlim(*x_range)

    ax.set_ylim(0, 1)
