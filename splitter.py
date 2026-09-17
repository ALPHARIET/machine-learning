import os

base_dir = r"d:\download"

common_imports = """import numpy as np
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

"""

ch2_base = """iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y
)
"""

ch2_scaled = ch2_base + """
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""

ch2_reg = """rng = np.random.RandomState(0)
X_reg = rng.rand(100, 1) * 10
y_reg = 2 * X_reg[:, 0] + 1 + rng.randn(100)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, random_state=0
)
"""

ch3_blob = """X_blob, y_blob = make_blobs(
    n_samples=200,
    centers=3,
    random_state=0,
    cluster_std=1.2
)
"""

ch3_cancer = """cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, random_state=0
)

scaler_cancer = StandardScaler()
X_train_c_scaled = scaler_cancer.fit_transform(X_train_c)
X_test_c_scaled = scaler_cancer.transform(X_test_c)
"""

ch3_digits = """digits = load_digits()
X_digits = digits.data
y_digits = digits.target

scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits)
"""

ch3_moons = """X_moons, y_moons = make_moons(
    n_samples=200,
    noise=0.05,
    random_state=0
)
"""

files_to_create = {
    "ch2_1_dataset.py": (ch2_base, """print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test :", X_test.shape)
"""),
    "ch2_2_knn.py": (ch2_base, """knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("\\n=== KNN ===")
print("Akurasi:", knn.score(X_test, y_test))
print("Prediksi:", y_pred_knn[:10])
"""),
    "ch2_3_logistic_regression.py": (ch2_base, """logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

print("\\n=== Logistic Regression ===")
print("Akurasi:", logreg.score(X_test, y_test))
"""),
    "ch2_4_naive_bayes.py": (ch2_base, """naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

print("\\n=== Naive Bayes ===")
print("Akurasi:", naive_bayes.score(X_test, y_test))
"""),
    "ch2_5_decision_tree.py": (ch2_base, """tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

print("\\n=== Decision Tree ===")
print("Akurasi:", tree.score(X_test, y_test))

plt.figure(figsize=(14, 8))
plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=list(iris.target_names),
    filled=True
)
plt.title("Decision Tree pada Dataset Iris")
plt.show()
"""),
    "ch2_6_random_forest.py": (ch2_base, """forest = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)
forest.fit(X_train, y_train)

print("\\n=== Random Forest ===")
print("Akurasi:", forest.score(X_test, y_test))

print("Feature importance:")
for name, importance in zip(iris.feature_names, forest.feature_importances_):
    print(f"{name}: {importance:.4f}")
"""),
    "ch2_7_svm.py": (ch2_scaled, """svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train_scaled, y_train)

print("\\n=== SVM ===")
print("Akurasi:", svm.score(X_test_scaled, y_test))
"""),
    "ch2_8_mlp.py": (ch2_scaled, """mlp = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=0
)
mlp.fit(X_train_scaled, y_train)

print("\\n=== Neural Network / MLP ===")
print("Akurasi:", mlp.score(X_test_scaled, y_test))
"""),
    "ch2_9_perbandingan_model.py": (ch2_scaled, """models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM": SVC()
}

print("\\n=== PERBANDINGAN MODEL ===")
for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

    print(f"{name}: {score:.4f}")
"""),
    "ch2_10_linear_regression.py": (ch2_reg, """linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

pred_reg = linear_model.predict(X_test_reg)

print("\\n=== Linear Regression ===")
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
"""),
    "ch2_11_ridge_regression.py": (ch2_reg, """ridge = Ridge(alpha=1.0)
ridge.fit(X_train_reg, y_train_reg)

print("\\n=== Ridge Regression ===")
print("RMSE:", np.sqrt(
    mean_squared_error(y_test_reg, ridge.predict(X_test_reg))
))
"""),
    "ch3_1_dataset_2d.py": (ch3_blob, """plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob)
plt.title("Dataset make_blobs")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
"""),
    "ch3_2_standard_scaler.py": (ch3_blob, """scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X_blob)

print("\\n=== StandardScaler ===")
print("Rata-rata setelah scaling:", X_standard.mean(axis=0))
print("Standar deviasi:", X_standard.std(axis=0))
"""),
    "ch3_3_minmax_scaler.py": (ch3_blob, """scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_blob)

print("\\n=== MinMaxScaler ===")
print("Nilai minimum:", X_minmax.min(axis=0))
print("Nilai maksimum:", X_minmax.max(axis=0))
"""),
    "ch3_4_robust_scaler.py": (ch3_blob, """scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_blob)

print("\\n=== RobustScaler ===")
print("Contoh data:", X_robust[:5])
"""),
    "ch3_5_normalizer.py": (ch3_blob, """normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X_blob)

print("\\n=== Normalizer ===")
print("Panjang vektor pertama:", np.linalg.norm(X_normalized[0]))
"""),
    "ch3_6_scaling_knn.py": (ch3_cancer, """knn_without_scaling = KNeighborsClassifier(n_neighbors=3)
knn_without_scaling.fit(X_train_c, y_train_c)

knn_with_scaling = KNeighborsClassifier(n_neighbors=3)
knn_with_scaling.fit(X_train_c_scaled, y_train_c)

print("\\n=== Pengaruh Scaling ===")
print("Tanpa scaling:", knn_without_scaling.score(X_test_c, y_test_c))
print("Dengan scaling:", knn_with_scaling.score(X_test_c_scaled, y_test_c))
"""),
    "ch3_7_pca.py": (ch3_digits, """pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits_scaled)

print("\\n=== PCA ===")
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
"""),
    "ch3_8_nmf.py": (ch3_digits, """# NMF membutuhkan data yang tidak negatif.
X_digits_nonnegative = digits.data

nmf = NMF(n_components=10, init="nndsvda", random_state=0, max_iter=500)
X_nmf = nmf.fit_transform(X_digits_nonnegative)

print("\\n=== NMF ===")
print("Ukuran data setelah NMF:", X_nmf.shape)
"""),
    "ch3_9_tsne.py": (ch3_digits, """# t-SNE dapat membutuhkan waktu lebih lama pada dataset besar.
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
"""),
    "ch3_10_kmeans.py": (ch3_blob, """kmeans = KMeans(
    n_clusters=3,
    random_state=0,
    n_init=10
)

cluster_kmeans = kmeans.fit_predict(X_blob)

print("\\n=== K-Means ===")
print("Label cluster:", np.unique(cluster_kmeans))
print("Silhouette score:", silhouette_score(X_blob, cluster_kmeans))

plt.scatter(X_blob[:, 0], X_blob[:, 1], c=cluster_kmeans)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="*",
    s=250
)
plt.title("K-Means Clustering")
plt.show()
"""),
    "ch3_11_agglomerative.py": (ch3_blob, """agglo = AgglomerativeClustering(n_clusters=3)
cluster_agglo = agglo.fit_predict(X_blob)

print("\\n=== Agglomerative Clustering ===")
print("Silhouette score:", silhouette_score(X_blob, cluster_agglo))

plt.scatter(X_blob[:, 0], X_blob[:, 1], c=cluster_agglo)
plt.title("Agglomerative Clustering")
plt.show()
"""),
    "ch3_12_dbscan.py": (ch3_moons, """dbscan = DBSCAN(eps=0.3, min_samples=5)
cluster_dbscan = dbscan.fit_predict(X_moons)

print("\\n=== DBSCAN ===")
print("Label cluster:", np.unique(cluster_dbscan))

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=cluster_dbscan)
plt.title("DBSCAN pada Dataset Two Moons")
plt.show()
"""),
    "ch3_13_perbandingan_clustering.py": (ch3_moons, """X_moons_scaled = StandardScaler().fit_transform(X_moons)

kmeans_moons = KMeans(n_clusters=2, random_state=0, n_init=10)
agglo_moons = AgglomerativeClustering(n_clusters=2)
dbscan_moons = DBSCAN(eps=0.3, min_samples=5)

labels_kmeans = kmeans_moons.fit_predict(X_moons_scaled)
labels_agglo = agglo_moons.fit_predict(X_moons_scaled)
labels_dbscan = dbscan_moons.fit_predict(X_moons_scaled)

print("\\n=== PERBANDINGAN CLUSTERING ===")
print("K-Means ARI:", adjusted_rand_score(y_moons, labels_kmeans))
print("Agglomerative ARI:", adjusted_rand_score(y_moons, labels_agglo))
print("DBSCAN ARI:", adjusted_rand_score(y_moons, labels_dbscan))
""")
}

for filename, (data_setup, specific_code) in files_to_create.items():
    filepath = os.path.join(base_dir, filename)
    with open(filepath, "w") as f:
        f.write(common_imports + "\n")
        f.write(data_setup + "\n")
        f.write(specific_code + "\n")

print("Files created successfully.")
