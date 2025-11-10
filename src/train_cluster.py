# src/train_cluster.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def find_elbow_inertia(X, max_k=8):
    """Mengembalikan list inertias untuk k=2..max_k (untuk Elbow plot jika ingin)."""
    inertias = []
    ks = list(range(2, max_k+1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return ks, inertias

def train_kmeans(X, n_clusters=3):
    """Latih KMeans dan kembalikan model, labels, silhouette score."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    return km, labels, sil

def pca_2d_projection(X):
    """Proyeksikan X ke 2D menggunakan PCA (untuk plotting)."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    return X2, pca

def plot_clusters_2d(X2, labels, savepath=None):
    """Buat scatter plot 2D hasil PCA."""
    plt.figure(figsize=(8,6))
    unique = sorted(set(labels))
    for u in unique:
        mask = labels == u
        plt.scatter(X2[mask,0], X2[mask,1], label=f'cluster {u}', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('Clusters (PCA 2D)')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()
