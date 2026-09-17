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


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y
)

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)
forest.fit(X_train, y_train)

print("\n=== Random Forest ===")
print("Akurasi:", forest.score(X_test, y_test))

print("Feature importance:")
for name, importance in zip(iris.feature_names, forest.feature_importances_):
    print(f"{name}: {importance:.4f}")

