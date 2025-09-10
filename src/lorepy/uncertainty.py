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
    """
    Estimates uncertainty in model predictions using resampling or jackknife methods.

    :param x: Name of the feature variable to analyze.
    :param X_reg: Feature matrix for regression/classification.
    :param y_reg: Target variable.
    :param x_range: Tuple (min, max) specifying the range of values for the feature variable `x` to evaluate.
    :param mode: Method for uncertainty estimation. Either "resample" (bootstrap) or "jackknife".
    :param jackknife_fraction: Fraction of data to keep in each jackknife iteration (only used if mode="jackknife").
    :param iterations: Number of resampling or jackknife iterations.
    :param confounders: List of tuples (feature, reference value) pairs representing confounder features and their reference values.
    :param clf: Classifier to use for fitting. If None, uses LogisticRegression.
    :return: Tuple containing output DataFrame with aggregated uncertainty statistics and long_df DataFrame with all resampled predictions.
    """
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


def _get_feature_importance(
    x: str,
    X_reg,
    y_reg,
    mode="resample",
    jackknife_fraction: float = 0.8,
    iterations: int = 100,
    confounders=None,
    clf=None,
):
    """
    Estimates the importance of the x-feature in predicting class labels using permutation-based
    feature importance with resampling or jackknife methods. Uses accuracy as the performance metric.

    :param x: Name of the feature variable to analyze for importance.
    :param X_reg: Feature matrix for regression/classification.
    :param y_reg: Target variable.
    :param mode: Method for uncertainty estimation. Either "resample" (bootstrap) or "jackknife".
    :param jackknife_fraction: Fraction of data to keep in each jackknife iteration (only used if mode="jackknife").
    :param iterations: Number of resampling or jackknife iterations.
    :param confounders: List of tuples (feature, reference value) pairs representing confounder features and their reference values.
    :param clf: Classifier to use for fitting. If None, uses LogisticRegression.
    :return: Dictionary containing feature importance statistics including mean importance, confidence intervals, and significance metrics.
    """
    confounders = [] if confounders is None else confounders

    importance_scores = []

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

        # Fit model with original data
        lg_normal = LogisticRegression() if clf is None else clf
        lg_normal.fit(X_keep, y_keep)
        normal_accuracy = lg_normal.score(X_keep, y_keep)

        # Create shuffled version by permuting only the x-feature (first column)
        # Keep confounders intact since they represent controlled variables
        X_shuffled = X_keep.copy()
        X_shuffled[:, 0] = np.random.permutation(X_shuffled[:, 0])

        # Fit model with shuffled x-feature
        lg_shuffled = LogisticRegression() if clf is None else clf
        lg_shuffled.fit(X_shuffled, y_keep)
        shuffled_accuracy = lg_shuffled.score(X_shuffled, y_keep)

        # Feature importance = performance drop when x-feature is shuffled
        importance = normal_accuracy - shuffled_accuracy
        importance_scores.append(importance)

    importance_scores = np.array(importance_scores)

    # Calculate statistics
    mean_importance = np.mean(importance_scores)
    std_importance = np.std(importance_scores)
    ci_95_low = np.percentile(importance_scores, 2.5)
    ci_95_high = np.percentile(importance_scores, 97.5)

    # Significance metrics
    significant_positive = np.sum(importance_scores > 0) / iterations
    significant_negative = np.sum(importance_scores < 0) / iterations

    # Empirical p-value (two-tailed test)
    p_value = (
        2 * min(significant_positive, significant_negative)
        if significant_positive != significant_negative
        else 1.0
    )

    return {
        "feature": x,
        "mean_importance": mean_importance,
        "std_importance": std_importance,
        "importance_95ci_low": ci_95_low,
        "importance_95ci_high": ci_95_high,
        "proportion_positive": significant_positive,
        "proportion_negative": significant_negative,
        "p_value": p_value,
        "iterations": iterations,
        "mode": mode,
        "interpretation": f"Feature importance: {mean_importance:.4f} ± {std_importance:.4f}. "
        f"Positive in {significant_positive:.1%} of iterations (p={p_value:.4f})",
    }


def uncertainty_plot(
    data: DataFrame,
    x: str,
    y: str,
    x_range=None,
    mode="resample",
    jackknife_fraction=0.8,
    iterations=100,
    confounders=None,
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
    confounders = [] if confounders is None else confounders

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


def feature_importance(
    data: DataFrame,
    x: str,
    y: str,
    mode="resample",
    jackknife_fraction=0.8,
    iterations=100,
    confounders=None,
    clf=None,
):
    """
    Estimates the importance of a feature in predicting class labels using permutation-based
    feature importance with resampling or jackknife methods. Uses accuracy as the performance metric.

    This function provides statistical assessment of whether the x-feature is significantly
    associated with the class distribution (y-variable). Higher importance scores indicate
    stronger predictive relationships.

    :param data: The input dataframe containing all features and target variable.
    :param x: The name of the feature to analyze for importance.
    :param y: The name of the target variable.
    :param mode: Method for uncertainty estimation. Either "resample" (bootstrap) or "jackknife".
    :param jackknife_fraction: Fraction of data to keep in each jackknife iteration (only used if mode="jackknife").
    :param iterations: Number of resampling or jackknife iterations.
    :param confounders: List of tuples (feature, reference value) pairs representing confounder features and their reference values.
    :param clf: Classifier to use for fitting. If None, uses LogisticRegression.
    :return: Dictionary containing feature importance statistics including mean importance, confidence intervals, and significance metrics.

    Example:
        >>> import pandas as pd
        >>> from lorepy import feature_importance
        >>> from sklearn.datasets import load_iris
        >>>
        >>> iris = load_iris()
        >>> df = pd.DataFrame(iris.data, columns=iris.feature_names)
        >>> df['species'] = iris.target
        >>>
        >>> stats = feature_importance(df, x='sepal length (cm)', y='species')
        >>> print(stats['interpretation'])
        'Feature importance: 0.234 ± 0.045. Positive in 98.0% of iterations (p=0.020)'
    """
    confounders = [] if confounders is None else confounders

    # Prepare data using existing helper function
    X_reg, y_reg, _ = _prepare_data(data, x, y, confounders)

    # Call internal function to do the heavy lifting
    return _get_feature_importance(
        x=x,
        X_reg=X_reg,
        y_reg=y_reg,
        mode=mode,
        jackknife_fraction=jackknife_fraction,
        iterations=iterations,
        confounders=confounders,
        clf=clf,
    )
