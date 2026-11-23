from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KOLOM_INPUT_MENTAH = ["date", "precipitation", "temp_max", "temp_min", "wind"]
KOLOM_TARGET = "weather"

KOLOM_NUMERIK = [
    "precipitation",
    "log_precipitation",
    "temp_max",
    "temp_min",
    "wind",
    "temp_range",
    "temp_mean",
    "is_raining",
    "is_freezing",
    "month",
    "day_of_year",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

KOLOM_KATEGORIK = ["season"]


def tentukan_musim(bulan: int) -> str:
    """Memetakan bulan (1-12) ke empat musim di belahan bumi utara (Seattle, AS)."""
    if bulan in (12, 1, 2):
        return "winter"
    elif bulan in (3, 4, 5):
        return "spring"
    elif bulan in (6, 7, 8):
        return "summer"
    return "autumn"


def audit_data(df: pd.DataFrame) -> dict:
    """Melakukan audit dasar pada dataset mentah Seattle Weather."""
    tanggal = pd.to_datetime(df["date"], errors="coerce")
    return {
        "jumlah_baris": int(len(df)),
        "jumlah_kolom": int(df.shape[1]),
        "tipe_data": df.dtypes.astype(str).to_dict(),
        "missing_value": df.isnull().sum().to_dict(),
        "baris_duplikat": int(df.duplicated().sum()),
        "tanggal_duplikat": int(df["date"].duplicated().sum()),
        "tanggal_tidak_valid": int(tanggal.isna().sum()),
        "temp_max_kurang_dari_temp_min": int((df["temp_max"] < df["temp_min"]).sum()),
        "precipitation_negatif": int((df["precipitation"] < 0).sum()),
        "wind_negatif": int((df["wind"] < 0).sum()),
        "distribusi_target": df[KOLOM_TARGET].value_counts().to_dict()
        if KOLOM_TARGET in df.columns
        else {},
    }


def rekayasa_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fungsi feature engineering row-wise (stateless) sehingga aman dari data leakage.
    - Membuat fitur wajib dari date: month, day_of_year, season (+ fitur siklis sin/cos).
    - Membuat fitur turunan cuaca: temp_range, temp_mean, is_raining, is_freezing, log_precipitation.
    - Membuang kolom date mentah (unik per baris) dan tidak mengekstrak year (pergeseran distribusi label).
    """
    hasil = df.copy()

    tanggal = pd.to_datetime(hasil["date"], errors="coerce")

    # 1. Fitur waktu wajib
    hasil["month"] = tanggal.dt.month
    hasil["day_of_year"] = tanggal.dt.dayofyear
    hasil["season"] = hasil["month"].apply(
        lambda m: tentukan_musim(int(m)) if pd.notna(m) else np.nan
    )

    # 2. Fitur siklis agar akhir tahun (Desember) berdekatan dengan awal tahun (Januari)
    hasil["month_sin"] = np.sin(2 * np.pi * hasil["month"] / 12.0)
    hasil["month_cos"] = np.cos(2 * np.pi * hasil["month"] / 12.0)
    hasil["doy_sin"] = np.sin(2 * np.pi * hasil["day_of_year"] / 366.0)
    hasil["doy_cos"] = np.cos(2 * np.pi * hasil["day_of_year"] / 366.0)

    # 3. Fitur turunan berbasis domain cuaca
    hasil["temp_range"] = hasil["temp_max"] - hasil["temp_min"]
    hasil["temp_mean"] = (hasil["temp_max"] + hasil["temp_min"]) / 2.0
    hasil["is_raining"] = (hasil["precipitation"] > 0).astype(int)
    hasil["is_freezing"] = (hasil["temp_min"] <= 0).astype(int)
    hasil["log_precipitation"] = np.log1p(hasil["precipitation"].clip(lower=0))

    kolom_akhir = KOLOM_NUMERIK + KOLOM_KATEGORIK
    return hasil[kolom_akhir]


class SeattleFeatureEngineer(BaseEstimator, TransformerMixin):
    """Wrapper scikit-learn untuk fungsi rekayasa_fitur agar bisa masuk ke Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=KOLOM_INPUT_MENTAH)
        return rekayasa_fitur(X)


def buat_preprocessor() -> ColumnTransformer:
    """
    Membuat ColumnTransformer untuk imputasi, scaling, dan one-hot encoding.
    Statistik preprocessing hanya dihitung dari data train saat pipeline di-.fit().
    """
    pipa_numerik = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipa_kategorik = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", pipa_numerik, KOLOM_NUMERIK),
            ("cat", pipa_kategorik, KOLOM_KATEGORIK),
        ]
    )


if __name__ == "__main__":
    path_data = Path("dataset") / "kasus05_seattle_weather.csv"
    if not path_data.exists():
        path_data = Path("kasus05_seattle_weather.csv")

    df_mentah = pd.read_csv(path_data)
    hasil_audit = audit_data(df_mentah)

    print("=== HASIL AUDIT DATA ===")
    for k, v in hasil_audit.items():
        print(f"{k}: {v}")

    df_fitur = rekayasa_fitur(df_mentah[KOLOM_INPUT_MENTAH])
    print("\n=== CONTOH HASIL FEATURE ENGINEERING (5 BARIS PERTAMA) ===")
    print("Ukuran fitur:", df_fitur.shape)
    print(df_fitur.head())