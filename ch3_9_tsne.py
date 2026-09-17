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

# t-SNE dapat membutuhkan waktu lebih lama pada dataset besar.
tsne = TSNE(
    n_components=2,
    random_state=0,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X_digits_scaled[:500])

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_digits[:500],
    cmap="tab10",
    s=15
)
plt.colorbar(scatter)
plt.title("t-SNE Dataset Digits")
plt.show()

