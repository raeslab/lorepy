import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


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


def _get_dots_df(X, y, lg, y_feature, confounders=[]) -> DataFrame:
    output = []

    for v, s in zip(X, y):
        proba = lg.predict_proba([v] + [i[1] for i in confounders])
        i = list(lg.classes_).index(s)
        min_value = sum(proba[0][:i])
        max_value = sum(proba[0][: i + 1])
        margin = (max_value - min_value) / 10
        ypos = np.random.uniform(low=min_value + margin, high=max_value - margin)
        output.append({y_feature: s, "x": v[0], "y": ypos})

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
    **kwargs
):
    """
    Code to create a loreplot with a numerical feature on the x-axis and categorical y from a pandas dataset

    :param data: Pandas dataframe with data
    :param x: Needs to be a numerical feature
    :param y: Categorical feature
    :param add_dots: Shows where true samples are in the plot (cannot be enabled when deconfounding for additional variables)
    :param x_range: Either None (range will be selected automatically) or a tuple with min and max value for the x-axis
    :param scatter_kws: Dictionary with keyword arguments to pass to the scatter function
    :param ax: subplot to draw on, in case lorepy is used in a subplot
    :param clf: provide a different scikit-learn classifier for the function. Should implement the predict_proba() and fit()
    :param confounders: list of tuples with the feature and reference value e.g. [("BMI", 25)] will confounders BMI and use a reference of 25 for plots
    :param kwargs: Additional arguments to pass to pandas' plot.area function
    """
    if ax is None:
        ax = plt.gca()

    x_features = [x] + [i[0] for i in confounders]

    tmp_df = data[x_features + [y]].dropna()
    X_reg = np.array(tmp_df[x_features])
    y_reg = np.array(tmp_df[y])

    if x_range is None:
        x_range = (X_reg[:, 0].min(), X_reg[:, 0].max())

    lg = LogisticRegression(multi_class="multinomial") if clf is None else clf
    lg.fit(X_reg, y_reg)

    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = "None"

    area_df = _get_area_df(lg, x, x_range, confounders=confounders)
    area_df.plot.area(ax=ax, **kwargs)

    if add_dots and len(confounders) == 0:
        dot_df = _get_dots_df(X_reg, y_reg, lg, y)
        if "color" not in scatter_kws.keys():
            scatter_kws["color"] = "w"
        if "alpha" not in scatter_kws.keys():
            scatter_kws["alpha"] = 0.3
        ax.scatter(dot_df["x"], dot_df["y"], **scatter_kws)

    ax.set_xlim(*x_range)

    ax.set_ylim(0, 1)


def uncertainty_plot(data: DataFrame,
    x: str,
    y: str, jackknife_fraction=0.8, jackknife_iterations=100):

    x_features = [x]

    tmp_df = data[x_features + [y]].dropna()
    X_reg = np.array(tmp_df[x_features])
    y_reg = np.array(tmp_df[y])

    x_range = (X_reg[:, 0].min(), X_reg[:, 0].max())

    areas = []
    for i in range(jackknife_iterations):
        X_keep, _, y_keep, _ = train_test_split(X_reg, y_reg, train_size=jackknife_fraction)

        lg = LogisticRegression(multi_class="multinomial")
        lg.fit(X_keep, y_keep)
        new_area = _get_area_df(lg, x, x_range).reset_index()
        # new_area["iteration"] = i + 1
        areas.append(new_area)

    long_df = pd.concat(areas).melt(id_vars=[x]).sort_values(x)

    # output = long_df.groupby([x, "variable"]).agg(
    #     pd.NamedAgg(min=min))

    return long_df
