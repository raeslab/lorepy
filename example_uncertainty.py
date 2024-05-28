from src.lorepy import uncertainty_plot

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

# Load iris dataset and convert to dataframe
iris_obj = load_iris()
iris_df = pd.DataFrame(iris_obj.data, columns=iris_obj.feature_names)

iris_df["species"] = [iris_obj.target_names[s] for s in iris_obj.target]

test = uncertainty_plot(data=iris_df, x="sepal width (cm)", y="species", jackknife_iterations=2)
print(test)