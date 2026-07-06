import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("glass.csv")

# Features and target
X = data.drop("Type", axis=1)
y = data["Type"]

# Normalize data
X = StandardScaler().fit_transform(X)

# 70-30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# KNN with Euclidean distance
knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn1.fit(X_train, y_train)
pred1 = knn1.predict(X_test)

# KNN with Manhattan distance
knn2 = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn2.fit(X_train, y_train)
pred2 = knn2.predict(X_test)

print("Euclidean Accuracy:",
      accuracy_score(y_test, pred1))

print("Manhattan Accuracy:",
      accuracy_score(y_test, pred2))