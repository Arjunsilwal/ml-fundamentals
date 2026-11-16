from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_bc, y_bc = load_breast_cancer(return_X_y=True)       
X_bc_scaled = StandardScaler().fit_transform(X_bc)

pca = PCA(n_components=2).fit(X_bc_scaled)
print(pca.explained_variance_ratio_.round(3))           

pca95 = PCA(n_components=0.95).fit(X_bc_scaled)
print(pca95.n_components_)                             