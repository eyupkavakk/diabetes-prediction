import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_inspect_data(file_path):
 
    try:
        df = pd.read_csv(file_path)
        print("Veri Seti Yüklendi.")
    except FileNotFoundError:
        print(f"Hata: '{file_path}' dosya yolu bulunamadı.")
        return None

    print("\nİlk 5 Satır:")
    print(df.head())

    print("\nVeri Tipleri:")
    print(df.dtypes)

    print("\nİstatistiksel Özet:")
    print(df.describe().T)

    print("\nVeri Seti Bilgisi:")
    df.info()

    print("\nEksik Değerler:")
    print(df.isnull().sum())

    return df

def detect_outliers(df, features):
   
    outlier_indices = []

    for col in features:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            Q1 = np.percentile(df[col].dropna(), 25)
            Q3 = np.percentile(df[col].dropna(), 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            lower_bound = Q1 - outlier_step
            upper_bound = Q3 + outlier_step

            outlier_cols = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.extend(outlier_cols)

    return list(set(outlier_indices))  # Tekrar eden indeksleri kaldır


def preprocess_data(df, target_col, outlier_features=None, test_size=0.2, random_state=42):
  
    if outlier_features:
        outlier_indices = detect_outliers(df, outlier_features)
        df_cleaned = df.drop(outlier_indices)
        print(f"{len(outlier_indices)} aykırı değer kaldırıldı.")
    else:
        df_cleaned = df.copy()

    X = df_cleaned.drop(target_col, axis=1).values
    y = df_cleaned[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test