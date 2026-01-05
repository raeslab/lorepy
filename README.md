[![Run Pytest](https://github.com/raeslab/lorepy/actions/workflows/autopytest.yml/badge.svg)](https://github.com/raeslab/lorepy/actions/workflows/autopytest.yml) [![Coverage](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/coverage-badge.svg)](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/coverage-badge.svg) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![DOI](https://zenodo.org/badge/686018963.svg)](https://zenodo.org/badge/latestdoi/686018963) [![PyPI version](https://badge.fury.io/py/lorepy.svg)](https://badge.fury.io/py/lorepy) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# lorepy: Logistic Regression Plots for Python

Logistic Regression plots are used to plot the distribution of a categorical dependent variable in function of a 
continuous independent variable.

If you prefer an R implementation of this package, have a look at [loreplotr](https://github.com/raeslab/loreplotr).

![LoRePlot example on Iris Dataset](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot.png)

## Why use lorepy ?

Lorepy offers distinct advantages over traditional methods like stacked bar plots. By employing a linear model, Lorepy 
captures overall trends across the entire feature range. It avoids arbitrary cut-offs and segmentation, enabling the 
visualization of uncertainty throughout the data range.

You can find examples of the Iris data visualized using stacked bar plots [here](https://github.com/raeslab/lorepy/blob/main/docs/lorepy_vs_bar_plots.md) for comparison.

## Installation

Lorepy can be installed using pip using the command below.

```
pip install lorepy
```

**This might take a few minutes as several dependencies need to be installed in the environment as well (e.g. pandas and scikit-learn)**

## Usage

Data needs to be provided as a DataFrame and the columns for the x (independent continuous) and y (dependant categorical)
variables need to be defined. Here the iris dataset is loaded and converted to an appropriate DataFrame. Once the data
is in shape it can be plotted using a single line of code ```loreplot(data=iris_df, x="sepal width (cm)", y="species")```.

```python
from lorepy import loreplot

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

iris_obj = load_iris()
iris_df = pd.DataFrame(iris_obj.data, columns=iris_obj.feature_names)

iris_df["species"] = [iris_obj.target_names[s] for s in iris_obj.target]

loreplot(data=iris_df, x="sepal width (cm)", y="species")

plt.show()
```

## Options

While lorepy has very few customizations, it is possible to pass arguments through to Pandas' 
[DataFrame.plot.area](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.area.html)
and Matplotlib's [pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) to change
the aesthetics of the plots.

**All examples below should run within a few seconds, though on large datasets the uncertainty plot can take a few minutes depending on the size of the dataset and settings selected.**

### Disable sample dots

Dots indicating where samples are located can be en-/disabled using the ```add_dots``` argument.

```python
loreplot(data=iris_df, x="sepal width (cm)", y="species", add_dots=False)
plt.show()
```

![LoRePlot dots can be disabled](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_no_dots.png)

### Custom styles

Additional keyword arguments are passed to Pandas' [DataFrame.plot.area](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.area.html).
This can be used, among other things, to define a custom colormap. For more options to customize these plots consult
Pandas' documentation.

```python
from matplotlib.colors import ListedColormap

colormap=ListedColormap(['red', 'green', 'blue'])

loreplot(data=iris_df, x="sepal width (cm)", y="species", colormap=colormap)
plt.show()
```
![LoRePlot custom colors](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_custom_color.png)


Using ```scatter_kws``` arguments for [pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)
can be set to change the appearance of the sample markers.

```python
scatter_options = {
    's': 20,                  # Marker size
    'alpha': 1,               # Fully opaque
    'color': 'black',         # Set color to black
    'marker': 'x'             # Set style to crosses
}

loreplot(data=iris_df, x="sepal width (cm)", y="species", scatter_kws=scatter_options)
plt.show()
```
![LoRePlot custom markers](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_custom_markers.png)

You can use LoRePlots in subplots as you would expect.

```python
fig, ax = plt.subplots(1,2, sharex=False, sharey=True)
loreplot(data=iris_df, x="sepal width (cm)", y="species", ax=ax[0])
loreplot(data=iris_df, x="petal width (cm)", y="species", ax=ax[1])

ax[0].get_legend().remove()
ax[0].set_title("Sepal Width")
ax[1].set_title("Petal Width")

plt.savefig('./docs/img/loreplot_subplot.png', dpi=150)
plt.show()
```

![LoRePlot in subplots](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_subplot.png)

By default lorepy uses a multi-class logistic regression model, however this can be replaced with any classifier
from scikit-learn that implements ```predict_proba``` and ```fit```. Below you can see the code and output with a
Support Vector Classifier (SVC) and Random Forest Classifier (RF).

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

fig, ax = plt.subplots(1, 2, sharex=False, sharey=True)

svc = SVC(probability=True)
rf = RandomForestClassifier(n_estimators=10, max_depth=2)

loreplot(data=iris_df, x="sepal width (cm)", y="species", clf=svc, ax=ax[0])
loreplot(data=iris_df, x="sepal width (cm)", y="species", clf=rf, ax=ax[1])

ax[0].get_legend().remove()
ax[0].set_title("SVC")
ax[1].set_title("RF")

plt.savefig("./docs/img/loreplot_other_clf.png", dpi=150)
plt.show()
```

![Lorepy with different types of classifiers](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_other_clf.png)


In case there are confounders, these can be taken into account using the ```confounders``` argument. This requires a
list of tuples, with the feature and the reference value for that feature to use in plots. E.g. if you wish to deconfound
for Body Mass Index (BMI) and use a BMI of 25 in plots, set this to [("BMI", 25)].

```python
loreplot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
    confounders=[("petal width (cm)", 1)],
)
plt.savefig("./docs/img/loreplot_confounder.png", dpi=150)
plt.show()
```

![Loreplot with a confounder](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_confounder.png)

In some cases the numerical feature on the x-axis isn't continuous (e.g. an integer number), this can lead to 
overplotting the dots. To avoid this to some extent a `jitter` feature is included, that adds some uniform noise to
the x-coordinates of the dots. The value specifies the range of the uniform noise added, the value of 0.05 in the 
example sets this range to [-0.05, 0.05].

```python
iris_df["sepal width (cm)"] = (
    np.round(iris_df["sepal width (cm)"] * 3) / 3
)  # Round values

loreplot(data=iris_df, x="sepal width (cm)", y="species", jitter=0.05)
plt.savefig("./docs/img/loreplot_jitter.png", dpi=150)
plt.show()
```
![Loreplot with a confounder](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/loreplot_jitter.png)

### Assess uncertainty

From loreplots it isn't possible to assess how certain we are of the prevalence of each group across the range. To
provide a view into this there is a function ```uncertainty_plot```, which can be used as shown below. This will use
```resampling``` (or ```jackknifing```) to determine the 50% and 95% interval of predicted values and show these in a
multi-panel plot with one plot per category.

```python
from lorepy import uncertainty_plot

uncertainty_plot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
)
plt.savefig("./docs/img/uncertainty_default.png", dpi=150)
plt.show()
```

![Default uncertainty plot](https://raw.githubusercontent.com/raeslab/lorepy/main/docs/img/uncertainty_default.png)

This also supports custom colors, ranges and classifiers. More examples are available in ```example_uncertainty.py```.

### Feature Importance Analysis

Lorepy provides statistical assessment of how strongly your x-feature is associated with the class distribution using the `feature_importance` function. This uses **permutation-based feature importance** to test whether the relationship you see in your loreplot is statistically significant.

#### How it Works

The function uses a robust resampling approach combined with sklearn's optimized permutation importance:

1. **Bootstrap/Jackknife Sampling**: Creates multiple subsamples of your data (default: 100 iterations)
2. **Permutation Importance**: For each subsample, uses sklearn's `permutation_importance` with proper cross-validation to avoid data leakage
3. **Feature Shuffling**: Randomly permutes the x-feature values while keeping confounders intact
4. **Performance Assessment**: Measures accuracy drop using statistically sound train/test splits
5. **Statistical Summary**: Aggregates results across all iterations to provide confidence intervals and significance testing

This approach works with **any sklearn classifier** (LogisticRegression, SVM, RandomForest, etc.) and properly handles confounders by keeping them constant during shuffling. The implementation uses sklearn's battle-tested permutation importance algorithm for reliable, unbiased results.

```python
from lorepy import feature_importance

# Basic usage
stats = feature_importance(data=iris_df, x="sepal width (cm)", y="species", iterations=100)
print(stats['interpretation'])
# Output: "Feature importance: 0.2019 ± 0.0433. Positive in 100.0% of iterations (p=0.0000)"
```

#### Understanding the Output

The function returns a dictionary with the following key statistics:

- **`mean_importance`**: Average accuracy drop when x-feature is shuffled (higher = more important)
- **`std_importance`**: Standard deviation of importance across iterations  
- **`importance_95ci_low/high`**: 95% confidence interval for the importance estimate
- **`proportion_positive`**: Fraction of iterations where importance > 0 (feature helps prediction)
- **`proportion_negative`**: Fraction of iterations where importance < 0 (feature hurts prediction)  
- **`p_value`**: Empirical p-value for statistical significance (< 0.05 typically considered significant)
- **`interpretation`**: Human-readable summary of the results

#### Advanced Usage

```python
from sklearn.svm import SVC

# With confounders and custom classifier
stats = feature_importance(
    data=data, 
    x="age", 
    y="disease",
    confounders=[("bmi", 25), ("sex", "female")],  # Control for these variables
    clf=SVC(probability=True),                      # Use SVM instead of logistic regression
    mode="jackknife",                              # Use jackknife instead of bootstrap
    iterations=200                                 # More iterations for precision
)

print(f"P-value: {stats['p_value']:.4f}")
print(f"95% CI: [{stats['importance_95ci_low']:.3f}, {stats['importance_95ci_high']:.3f}]")
```

#### Interpretation Guidelines

- **Strong Association**: `p_value < 0.01`, `proportion_positive > 95%`
- **Moderate Association**: `p_value < 0.05`, `proportion_positive > 80%`  
- **Weak/No Association**: `p_value > 0.05`, confidence interval includes zero
- **Negative Association**: `proportion_negative > proportion_positive` (unusual but possible)



## Development

Additional [documentation for developers](https://github.com/raeslab/lorepy/blob/main/docs/dev_docs.md) is included with details on running tests, building and deploying to PyPi.

## Contributing

Any contributions you make are **greatly appreciated**.

  * Found a bug or have some suggestions? Open an [issue](https://github.com/raeslab/lorepy/issues).
  * Pull requests are welcome! Though open an [issue](https://github.com/raeslab/lorepy/issues) first to discuss which features/changes you wish to implement.

## Contact

lorepy was developed by [Sebastian Proost](https://sebastian.proost.science/) at the 
[RaesLab](https://raeslab.sites.vib.be/en) and was based on R code written by 
[Sara Vieira-Silva](https://saravsilva.github.io/). As of version 0.2.0 lorepy is available under the 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) 
license. 

For commercial access inquiries, please contact [Jeroen Raes](mailto:jeroen.raes@kuleuven.vib.be).

