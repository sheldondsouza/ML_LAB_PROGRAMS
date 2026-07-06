import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Select features
x = data[:, 0]   # Sepal Length
y = data[:, 2]   # Petal Length

# Create scatter plot
plt.scatter(x, y, c=target, cmap="jet")

# Add labels and title
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot of Iris Dataset")

plt.show()