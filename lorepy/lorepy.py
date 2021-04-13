from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


def _get_area_df(X, lg, x_feature) -> DataFrame:
    values = np.linspace(X.min(), X.max(), num=200)
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
    scatter_kws: dict = dict({}),
    **kwargs
):
    """

    :param data:
    :param x:
    :param y:
    :param add_dots:
    :param scatter_kws:
    :param kwargs:
    :return:
    """
    tmp_df = data[[x, y]].dropna()
    X_reg = np.array(tmp_df[x]).reshape(-1, 1)
    y_reg = np.array(tmp_df[y])

    lg = LogisticRegression(multi_class="multinomial")
    lg.fit(X_reg, y_reg)

    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = 'None'

    area_df = _get_area_df(X_reg, lg, x)
    area_df.plot.area(**kwargs)

    if add_dots:
        dot_df = _get_dots_df(X_reg, y_reg, lg, y)
        if "color" not in scatter_kws.keys():
            scatter_kws["color"] = "w"
        if "alpha" not in scatter_kws.keys():
            scatter_kws["alpha"] = 0.3
        plt.scatter(dot_df["x"], dot_df["y"], **scatter_kws)

    plt.xlim(X_reg.min(), X_reg.max())
    plt.ylim(0, 1)
