import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data

x = data[:, 0]   # Sepal Length
y = data[:, 1]   # Sepal Width
z = data[:, 2]   # Petal Length

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap="jet")
ax.set_title("3D Surface Plot")

plt.show()