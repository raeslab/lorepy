from lorepy import uncertainty_plot

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import pandas as pd

# Load iris dataset and convert to dataframe
iris_obj = load_iris()
iris_df = pd.DataFrame(iris_obj.data, columns=iris_obj.feature_names)

iris_df["species"] = [iris_obj.target_names[s] for s in iris_obj.target]

# Default uncertainty plot
uncertainty_plot(data=iris_df, x="sepal width (cm)", y="species", iterations=100)
plt.savefig("./docs/img/uncertainty_default.png", dpi=150)
plt.show()

# Using jackknife instead of resample to assess uncertainty
uncertainty_plot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
    iterations=100,
    jackknife_fraction=0.8,
)
plt.savefig("./docs/img/uncertainty_jackknife.png", dpi=150)
plt.show()

# Uncertainty plot with custom colors


colormap = ListedColormap(["red", "green", "blue"])
uncertainty_plot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
    iterations=100,
    mode="resample",
    colormap=colormap,
)
plt.savefig("./docs/img/uncertainty_custom_color.png", dpi=150)
plt.show()

# Uncertainty plot with a confounder
uncertainty_plot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
    iterations=100,
    mode="resample",
    confounders=[("petal width (cm)", 1)],
)
plt.savefig("./docs/img/uncertainty_confounder.png", dpi=150)
plt.show()

# Uncertainty plot with a custom classifier
svc = SVC(probability=True)

uncertainty_plot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
    iterations=100,
    mode="resample",
    clf=svc,
)
plt.savefig("./docs/img/uncertainty_custom_classifier.png", dpi=150)
plt.show()
