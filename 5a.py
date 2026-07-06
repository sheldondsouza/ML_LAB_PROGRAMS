import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a Pandas DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Create the box plot
plt.figure(figsize=(8, 5))

sns.boxplot(data=data)

plt.title("Box Plot of Iris Dataset")

plt.show()