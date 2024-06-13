import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lorepy import loreplot
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load iris dataset and convert to dataframe
iris_obj = load_iris()
iris_df = pd.DataFrame(iris_obj.data, columns=iris_obj.feature_names)

iris_df["species"] = [iris_obj.target_names[s] for s in iris_obj.target]

# Basic Lore Plot with default style
loreplot(data=iris_df, x="sepal width (cm)", y="species")
plt.savefig("./docs/img/loreplot.png", dpi=150)
plt.show()

# Key word arguments (like colormap) can be passed to the DataFrame.plot.area

colormap = ListedColormap(["red", "green", "blue"])
loreplot(data=iris_df, x="sepal width (cm)", y="species", colormap=colormap)
plt.savefig("./docs/img/loreplot_custom_color.png", dpi=150)
plt.show()

# En-/disable sample markers with add_dots
loreplot(data=iris_df, x="sepal width (cm)", y="species", add_dots=False)
plt.savefig("./docs/img/loreplot_no_dots.png", dpi=150)
plt.show()

# Pass custom styles for markers using scatter_kws
scatter_options = {
    "s": 20,  # Marker size
    "alpha": 1,  # Fully opaque
    "color": "black",  # Set color to black
    "marker": "x",  # Set style to crosses
}

loreplot(data=iris_df, x="sepal width (cm)", y="species", scatter_kws=scatter_options)
plt.savefig("./docs/img/loreplot_custom_markers.png", dpi=150)
plt.show()

# Test in subplots

fig, ax = plt.subplots(1, 2, sharex=False, sharey=True)
loreplot(data=iris_df, x="sepal width (cm)", y="species", ax=ax[0])
loreplot(data=iris_df, x="petal width (cm)", y="species", ax=ax[1])

ax[0].get_legend().remove()
ax[0].set_title("Sepal Width")
ax[1].set_title("Petal Width")

plt.savefig("./docs/img/loreplot_subplot.png", dpi=150)
plt.show()

# Basic Lore Plot with default style but different classifier
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

# Basic Lore Plot with default style with one confounder
loreplot(
    data=iris_df,
    x="sepal width (cm)",
    y="species",
    confounders=[("petal width (cm)", 1)],
)
plt.savefig("./docs/img/loreplot_confounder.png", dpi=150)
plt.show()

# Basic Lore Plot with some jitter
iris_df["sepal width (cm)"] = (
    np.round(iris_df["sepal width (cm)"] * 3) / 3
)  # Round values

loreplot(data=iris_df, x="sepal width (cm)", y="species", jitter=0.05)
plt.savefig("./docs/img/loreplot_jitter.png", dpi=150)
plt.show()
