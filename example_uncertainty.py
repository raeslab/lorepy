from src.lorepy import uncertainty_plot

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

# Load iris dataset and convert to dataframe
iris_obj = load_iris()
iris_df = pd.DataFrame(iris_obj.data, columns=iris_obj.feature_names)

iris_df["species"] = [iris_obj.target_names[s] for s in iris_obj.target]

# uncertainty_plot(
#     data=iris_df, x="sepal width (cm)", y="species", iterations=100, jackknife_fraction=0.8
# )

# uncertainty_plot(
#     data=iris_df, x="sepal width (cm)", y="species", iterations=100, mode="resample"
# )

# from matplotlib.colors import ListedColormap
#
# colormap = ListedColormap(["red", "green", "blue"])
# uncertainty_plot(
#     data=iris_df, x="sepal width (cm)", y="species", iterations=100, mode="resample", colormap=colormap,
# )

# uncertainty_plot(
#     data=iris_df, x="sepal width (cm)", y="species", iterations=100, mode="resample", confounders=[("petal width (cm)", 1)]
# )

from sklearn.svm import SVC

svc = SVC(probability=True)

uncertainty_plot(
    data=iris_df, x="sepal width (cm)", y="species", iterations=100, mode="resample", clf=svc
)
plt.show()
