# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SDGs Country Clustering", layout="wide")
st.title("Clustering Negara Berdasarkan SDGs")
st.markdown("""
Aplikasi ini melakukan **clustering negara** berdasarkan indikator SDGs menggunakan **K-Means** dan **DBSCAN**.
""")

# ===== Upload Dataset =====
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV atau Excel (xls/xlsx)", 
    type=["csv","xls","xlsx"]
)

if uploaded_file is not None:
    try:
        # Membaca file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # ===== Exploratory Data Analysis (EDA) =====
        st.subheader("Exploratory Data Analysis (EDA)")
        st.markdown("**Histogram Fitur Numerik**")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        fig_hist, ax_hist = plt.subplots(len(numeric_cols)//3 +1, 3, figsize=(15,5*(len(numeric_cols)//3 +1)))
        ax_hist = ax_hist.flatten()
        for i, col in enumerate(numeric_cols):
            ax_hist[i].hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
            ax_hist[i].set_title(col)
        for j in range(i+1,len(ax_hist)):
            fig_hist.delaxes(ax_hist[j])
        st.pyplot(fig_hist)
        
        st.markdown("**Heatmap Korelasi Fitur Numerik**")
        fig_corr, ax_corr = plt.subplots(figsize=(12,10))
        sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
        
        # ===== Preprocessing =====
        st.sidebar.header("Preprocessing Options")
        missing_strategy = st.sidebar.selectbox("Handle Missing Values", ["median", "mean"])
        scale_data = st.sidebar.checkbox("Standardize Data", value=True)
        outlier_removal = st.sidebar.checkbox("Hapus Outlier (z-score > 3)", value=False)
        
        # Missing value
        strategy = 'median' if missing_strategy == "median" else 'mean'
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        
        # Outlier removal
        if outlier_removal:
            z_score = np.abs((df_imputed - df_imputed.mean())/df_imputed.std())
            df_imputed = df_imputed[(z_score < 3).all(axis=1)]
            st.warning(f"Data setelah outlier removal: {df_imputed.shape[0]} baris")
        
        # Scaling
        if scale_data:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
        else:
            df_scaled = df_imputed.copy()
        
        st.success("Preprocessing Selesai!")
        
        # ===== Clustering =====
        st.sidebar.header("K-Means Options")
        k = st.sidebar.slider("Jumlah cluster K-Means", min_value=2, max_value=10, value=4)
        
        st.sidebar.header("DBSCAN Options")
        eps = st.sidebar.slider("eps DBSCAN", 0.1, 10.0, 2.0)
        min_samples = st.sidebar.slider("min_samples DBSCAN", 1, 20, 5)
        
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)
        sil_score = silhouette_score(df_scaled, df['KMeans_Cluster'])
        
        st.subheader("Hasil Evaluasi K-Means")
        st.markdown(f"- **Silhouette Score:** {sil_score:.2f}")
        st.markdown("**Distribusi negara per cluster:**")
        cluster_counts_kmeans = df['KMeans_Cluster'].value_counts().sort_index()
        st.table(cluster_counts_kmeans)
        
        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)
        
        st.subheader("Hasil Evaluasi DBSCAN")
        st.markdown("**Distribusi negara per cluster (termasuk noise):**")
        cluster_counts_dbscan = df['DBSCAN_Cluster'].value_counts().sort_index()
        st.table(cluster_counts_dbscan)
        
        # ===== Summary Statistik Cluster =====
        st.subheader("Summary Statistik per Cluster (K-Means)")
        for cluster in sorted(df['KMeans_Cluster'].unique()):
            st.markdown(f"**Cluster {cluster}**")
            st.dataframe(df[df['KMeans_Cluster'] == cluster][numeric_cols].describe().T[['mean','std','min','max']])
        
        st.subheader("Summary Statistik per Cluster (DBSCAN)")
        dbscan_clusters = sorted(df['DBSCAN_Cluster'].unique())
        for cluster in dbscan_clusters:
            label = "Noise" if cluster == -1 else f"Cluster {cluster}"
            st.markdown(f"**{label}**")
            st.dataframe(df[df['DBSCAN_Cluster'] == cluster][numeric_cols].describe().T[['mean','std','min','max']])
        
        # ===== Visualisasi Cluster 2D =====
        st.subheader("Visualisasi Cluster 2D (PCA)")
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        # K-Means
        scatter1 = ax[0].scatter(df_pca[:,0], df_pca[:,1], c=df['KMeans_Cluster'], cmap='viridis')
        ax[0].set_title("K-Means Clustering")
        ax[0].set_xlabel("PCA 1")
        ax[0].set_ylabel("PCA 2")
        legend1 = ax[0].legend(*scatter1.legend_elements(), title="Cluster")
        ax[0].add_artist(legend1)
        # DBSCAN
        scatter2 = ax[1].scatter(df_pca[:,0], df_pca[:,1], c=df['DBSCAN_Cluster'], cmap='rainbow')
        ax[1].set_title("DBSCAN Clustering")
        ax[1].set_xlabel("PCA 1")
        ax[1].set_ylabel("PCA 2")
        legend2 = ax[1].legend(*scatter2.legend_elements(), title="Cluster")
        ax[1].add_artist(legend2)
        st.pyplot(fig)
        
        # ===== Download Hasil =====
        st.subheader("Download Hasil Clustering")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='sdg_clustering_results.csv',
            mime='text/csv'
        )
        
        st.info("""
        **Petunjuk Penggunaan:**
        1. Upload file CSV atau Excel berisi data negara dan indikator SDGs.
        2. Pilih opsi preprocessing (missing value, scaling, outlier removal).
        3. Atur parameter K-Means dan DBSCAN.
        4. Lihat hasil clustering, visualisasi, dan download CSV.
        """)
        
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
else:
    st.info("Silakan upload file CSV atau Excel terlebih dahulu.")
