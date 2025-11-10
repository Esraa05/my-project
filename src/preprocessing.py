# src/preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    """Membaca dataset dari Excel (.xlsx) atau CSV."""
    path = str(path)
    if path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def select_numeric(df):
    """Ambil hanya kolom numerik (float/int) - mengabaikan kolom non-numerik."""
    num_df = df.select_dtypes(include=['number']).copy()
    return num_df

def clean_and_impute(df_num, strategy='median'):
    """Tangani missing value: imputasi dengan median (default)."""
    imputer = SimpleImputer(strategy=strategy)
    arr = imputer.fit_transform(df_num)
    df_imputed = pd.DataFrame(arr, columns=df_num.columns, index=df_num.index)
    return df_imputed, imputer

def normalize(df_num):
    """Standarisasi / normalisasi (StandardScaler) dan kembalikan dataframe."""
    scaler = StandardScaler()
    arr = scaler.fit_transform(df_num)
    df_scaled = pd.DataFrame(arr, columns=df_num.columns, index=df_num.index)
    return df_scaled, scaler
