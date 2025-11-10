SDGs Country Clustering Web App

Aplikasi web interaktif untuk clustering negara berdasarkan indikator SDGs menggunakan algoritma K-Means dan DBSCAN, dengan visualisasi hasil clustering dan opsi preprocessing data.

Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis negara-negara berdasarkan berbagai indikator SDGs (Sustainable Development Goals) dan mengelompokkan negara dengan karakteristik serupa.
Fitur utama:

Preprocessing data: handle missing values, scaling, outlier removal

Clustering: K-Means dan DBSCAN

Evaluasi cluster: silhouette score, distribusi cluster

Visualisasi cluster 2D menggunakan PCA

Download hasil clustering dalam format CSV

Teknologi dan Tools

Python 3.10+

Streamlit – untuk membuat web app interaktif

Pandas, NumPy – manipulasi data

Scikit-learn – preprocessing, K-Means, DBSCAN, PCA

Matplotlib, Seaborn – visualisasi

Joblib – menyimpan model (opsional)

Instalasi

Clone repository ini:

git clone https://github.com/Esraa05/my-project.git
cd my-project


Buat virtual environment:

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac


Install dependensi:

pip install -r requirements.txt

Cara Menjalankan Aplikasi

Untuk menjalankan web app:

streamlit run app.py


Akses aplikasi di browser: http://localhost:8501

Untuk menjalankan script testing / preprocessing / clustering secara langsung:

python test_run.py

Fitur

Upload file CSV atau Excel berisi data negara dan indikator SDGs

Pilihan preprocessing: handle missing values, scaling, hapus outlier

Pilihan parameter K-Means dan DBSCAN

Lihat distribusi cluster dan summary statistik

Visualisasi cluster 2D menggunakan PCA

Download hasil clustering sebagai CSV

Struktur Folder
my-project/
│
├─ app.py              
├─ test_run.py         
├─ data/               
├─ src/                
├─ requirements.txt    
└─ README.md

Cara Penggunaan

Upload dataset CSV atau Excel

Pilih opsi preprocessing sesuai kebutuhan

Atur parameter K-Means (jumlah cluster) dan DBSCAN (eps, min_samples)

Lihat hasil clustering dan visualisasi

Download file CSV hasil clustering