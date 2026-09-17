import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_blobs,
    make_moons,
    load_digits
)
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score


rng = np.random.RandomState(0)
X_reg = rng.rand(100, 1) * 10
y_reg = 2 * X_reg[:, 0] + 1 + rng.randn(100)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, random_state=0
)

linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

pred_reg = linear_model.predict(X_test_reg)

print("\n=== Linear Regression ===")
print("Koefisien:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred_reg)))

plt.scatter(X_test_reg, y_test_reg, label="Data asli")
plt.plot(X_test_reg, pred_reg, label="Prediksi")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()

