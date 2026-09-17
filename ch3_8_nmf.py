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

# NMF membutuhkan data yang tidak negatif.
X_digits_nonnegative = digits.data

nmf = NMF(n_components=10, init="nndsvda", random_state=0, max_iter=500)
X_nmf = nmf.fit_transform(X_digits_nonnegative)

print("\n=== NMF ===")
print("Ukuran data setelah NMF:", X_nmf.shape)

