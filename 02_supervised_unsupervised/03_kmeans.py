from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X_iris, y_iris = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X_iris)      # K-Means uses distances, so scale first

for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(k, round(km.inertia_, 1), round(silhouette_score(X_scaled, km.labels_), 3))