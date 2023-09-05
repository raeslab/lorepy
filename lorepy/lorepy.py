from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def _get_area_df(X, lg, x_feature, x_range=None) -> DataFrame:
    values = (
        np.linspace(X.min(), X.max(), num=200)
        if x_range is None
        else np.linspace(x_range[0], x_range[1], num=200)
    )
    proba = lg.predict_proba(values.reshape(-1, 1))
    proba_df = DataFrame(proba, columns=lg.classes_)
    proba_df[x_feature] = values
    proba_df.set_index(x_feature, inplace=True)

    return proba_df


def _get_dots_df(X, y, lg, y_feature) -> DataFrame:
    output = []

    for v, s in zip(X, y):
        proba = lg.predict_proba([v])
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
    **kwargs
):
    """
    Code to create a loreplot with a numerical feature on the x-axis and categorical y from a pandas dataset

    :param data: Pandas dataframe with data
    :param x: Needs to be a numerical feature
    :param y: Categorical feature
    :param add_dots: Shows where true samples are in the plot
    :param x_range: Either None (range will be selected automatically) or a tuple with min and max value for the x-axis
    :param scatter_kws: Dictionary with keyword arguments to pass to the scatter function
    :param ax: subplot to draw on, in case lorepy is used in a subplot
    :param kwargs: Additional arguments to pass to pandas' plot.area function
    """
    if ax is None:
        ax = plt.gca()
    tmp_df = data[[x, y]].dropna()
    X_reg = np.array(tmp_df[x]).reshape(-1, 1)
    y_reg = np.array(tmp_df[y])

    lg = LogisticRegression(multi_class="multinomial")
    lg.fit(X_reg, y_reg)

    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = "None"

    area_df = _get_area_df(X_reg, lg, x, x_range=x_range)
    area_df.plot.area(ax=ax, **kwargs)

    if add_dots:
        dot_df = _get_dots_df(X_reg, y_reg, lg, y)
        if "color" not in scatter_kws.keys():
            scatter_kws["color"] = "w"
        if "alpha" not in scatter_kws.keys():
            scatter_kws["alpha"] = 0.3
        ax.scatter(dot_df["x"], dot_df["y"], **scatter_kws)

    if x_range is None:
        ax.set_xlim(X_reg.min(), X_reg.max())
    else:
        ax.set_xlim(*x_range)

    ax.set_ylim(0, 1)
