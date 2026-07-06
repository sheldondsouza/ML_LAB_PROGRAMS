import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a Pandas DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Heatmap of correlation matrix
sns.heatmap(data.corr(), cmap='jet', annot=True)

plt.title("Heatmap of Iris Dataset")
plt.show()