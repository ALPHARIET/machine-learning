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


X_moons, y_moons = make_moons(
    n_samples=200,
    noise=0.05,
    random_state=0
)

X_moons_scaled = StandardScaler().fit_transform(X_moons)

kmeans_moons = KMeans(n_clusters=2, random_state=0, n_init=10)
agglo_moons = AgglomerativeClustering(n_clusters=2)
dbscan_moons = DBSCAN(eps=0.3, min_samples=5)

labels_kmeans = kmeans_moons.fit_predict(X_moons_scaled)
labels_agglo = agglo_moons.fit_predict(X_moons_scaled)
labels_dbscan = dbscan_moons.fit_predict(X_moons_scaled)

print("\n=== PERBANDINGAN CLUSTERING ===")
print("K-Means ARI:", adjusted_rand_score(y_moons, labels_kmeans))
print("Agglomerative ARI:", adjusted_rand_score(y_moons, labels_agglo))
print("DBSCAN ARI:", adjusted_rand_score(y_moons, labels_dbscan))

