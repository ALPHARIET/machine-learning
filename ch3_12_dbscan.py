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

dbscan = DBSCAN(eps=0.3, min_samples=5)
cluster_dbscan = dbscan.fit_predict(X_moons)

print("\n=== DBSCAN ===")
print("Label cluster:", np.unique(cluster_dbscan))

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=cluster_dbscan)
plt.title("DBSCAN pada Dataset Two Moons")
plt.show()

