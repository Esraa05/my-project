# test_run.py
from src.preprocessing import load_dataset, select_numeric, clean_and_impute, normalize
from src.train_cluster import train_kmeans, pca_2d_projection, plot_clusters_2d
import os

DATA_PATH = os.path.join("data", "sdg_cluster_dataset_2021.csv.xlsx")

def main():
    print("1) Load dataset:", DATA_PATH)
    df = load_dataset(DATA_PATH)
    print(" - shape:", df.shape)
    print(" - columns:", list(df.columns))

    # Pilih kolom numerik
    df_num = select_numeric(df)
    print("2) Numeric columns used:", list(df_num.columns))
    print(" - numeric shape:", df_num.shape)

    # Imputasi missing dan normalisasi
    df_imputed, imputer = clean_and_impute(df_num, strategy='median')
    df_scaled, scaler = normalize(df_imputed)
    print("3) Imputasi & normalisasi selesai.")

    # Latih KMeans
    n_clusters = 4
    print(f"4) Melatih KMeans dengan n_clusters={n_clusters} ...")
    km, labels, sil = train_kmeans(df_scaled, n_clusters=n_clusters)
    print(f"   -> Silhouette score: {sil:.4f}")

    # Tampilkan ukuran cluster
    import pandas as pd
    df_out = df.copy()
    df_out['cluster'] = labels
    print("   -> Counts per cluster:")
    print(df_out['cluster'].value_counts().sort_index())

    # PCA + plot (akan tampil gambar)
    X2, pca = pca_2d_projection(df_scaled.values)
    print("5) Menampilkan plot PCA 2D (tutup window plot untuk melanjutkan).")
    plot_clusters_2d(X2, labels, savepath="cluster_pca.png")
    print("Plot tersimpan sebagai cluster_pca.png")

if __name__ == "__main__":
    main()
