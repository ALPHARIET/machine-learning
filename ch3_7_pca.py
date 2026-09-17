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


digits = load_digits()
X_digits = digits.data
y_digits = digits.target

scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits_scaled)

print("\n=== PCA ===")
print("Ukuran data sebelum PCA:", X_digits.shape)
print("Ukuran data sesudah PCA:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_digits,
    cmap="tab10",
    s=15
)
plt.colorbar(scatter)
plt.title("PCA Dataset Digits")
plt.xlabel("Komponen 1")
plt.ylabel("Komponen 2")
plt.show()

