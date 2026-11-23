from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from fitur import (
    KOLOM_INPUT_MENTAH,
    KOLOM_TARGET,
    SeattleFeatureEngineer,
    audit_data,
    buat_preprocessor,
)

PATH_DATASET = Path("dataset") / "kasus05_seattle_weather.csv"
PATH_FOLDER_MODEL = Path("models")
PATH_MODEL = PATH_FOLDER_MODEL / "model_kasus05.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def daftar_kandidat_model() -> dict:
    """
    Menyiapkan beberapa algoritma klasifikasi untuk dibandingkan lewat Cross-Validation.
    Menyertakan Logistic Regression default sebagai pembanding baseline terhadap
    strategi penanganan ketidakseimbangan kelas (class_weight='balanced').
    """
    return {
        "Logistic Regression (Default)": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees (Balanced)": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def bangun_pipeline(estimator) -> Pipeline:
    """Menggabungkan feature engineering, preprocessing, dan model dalam satu Pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineering", SeattleFeatureEngineer()),
            ("preprocessing", buat_preprocessor()),
            ("classifier", estimator),
        ]
    )


def main():
    path_data = PATH_DATASET
    if not path_data.exists():
        fallback = Path("kasus05_seattle_weather.csv")
        if fallback.exists():
            path_data = fallback
        else:
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di {PATH_DATASET}. "
                "Pastikan kasus05_seattle_weather.csv ada di folder dataset/."
            )

    df = pd.read_csv(path_data)
    hasil_audit = audit_data(df)

    X = df[KOLOM_INPUT_MENTAH].copy()
    y = df[KOLOM_TARGET].copy()

    # 1. Split train/test SEBELUM imputasi, scaling, atau encoding (anti data leakage)
    # Gunakan stratify=y agar proporsi kelas minoritas (snow, drizzle) terjaga
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=== PEMBAGIAN DATA (STRATIFIED SPLIT) ===")
    print(f"Data train: {len(X_train)} baris | Data test: {len(X_test)} baris")
    print("Distribusi train:", y_train.value_counts().to_dict())
    print("Distribusi test :", y_test.value_counts().to_dict())

    # 2. Pemilihan model terbaik HANYA menggunakan Cross-Validation di data train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "f1_weighted": "f1_weighted",
        "accuracy": "accuracy",
    }

    kandidat = daftar_kandidat_model()
    ringkasan_cv = []

    print("\n=== PERBANDINGAN MODEL (5-FOLD STRATIFIED CV DI DATA TRAIN) ===")
    for nama_model, estimator in kandidat.items():
        pipa = bangun_pipeline(estimator)
        skor = cross_validate(
            pipa, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        baris_hasil = {
            "model": nama_model,
            "cv_f1_macro_mean": float(np.mean(skor["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(skor["test_f1_macro"])),
            "cv_balanced_acc_mean": float(np.mean(skor["test_balanced_accuracy"])),
            "cv_f1_weighted_mean": float(np.mean(skor["test_f1_weighted"])),
            "cv_accuracy_mean": float(np.mean(skor["test_accuracy"])),
        }
        ringkasan_cv.append(baris_hasil)

    df_cv = pd.DataFrame(ringkasan_cv).sort_values(
        by="cv_f1_macro_mean", ascending=False
    )
    print(df_cv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Pilih model terbaik berdasarkan metrik utama Macro F1-Score
    nama_terbaik = df_cv.iloc[0]["model"]
    print(f"\nModel terpilih (CV F1-Macro tertinggi): {nama_terbaik}")

    # 3. Latih ulang model terbaik pada seluruh data train
    pipeline_terbaik = bangun_pipeline(kandidat[nama_terbaik])
    pipeline_terbaik.fit(X_train, y_train)

    # 4. Evaluasi pada data test TEPAT SATU KALI
    y_pred = pipeline_terbaik.predict(X_test)
    labels = list(pipeline_terbaik.classes_)

    metrik_test = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classification_report_dict": classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "classification_report_text": classification_report(
            y_test, y_pred, labels=labels, zero_division=0
        ),
    }

    print("\n=== HASIL EVALUASI FINAL PADA DATA TEST (DIEVALUASI 1 KALI) ===")
    print(f"Macro F1-Score    : {metrik_test['f1_macro']:.4f}")
    print(f"Balanced Accuracy : {metrik_test['balanced_accuracy']:.4f}")
    print(f"Weighted F1-Score : {metrik_test['f1_weighted']:.4f}")
    print(f"Accuracy          : {metrik_test['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrik_test["classification_report_text"])

    # 5. Simpan pipeline lengkap + metadata ke models/model_kasus05.joblib
    PATH_FOLDER_MODEL.mkdir(parents=True, exist_ok=True)
    artefak = {
        "pipeline": pipeline_terbaik,
        "nama_model_terbaik": nama_terbaik,
        "kolom_input_mentah": KOLOM_INPUT_MENTAH,
        "kolom_target": KOLOM_TARGET,
        "kelas_target": labels,
        "hasil_audit": hasil_audit,
        "hasil_cv": df_cv.to_dict(orient="records"),
        "metrik_test": metrik_test,
        "info_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "jumlah_train": int(len(X_train)),
            "jumlah_test": int(len(X_test)),
            "distribusi_train": y_train.value_counts().to_dict(),
            "distribusi_test": y_test.value_counts().to_dict(),
        },
        "strategi_imbalance": (
            "Stratified train/test split + Stratified 5-Fold CV + "
            "pembobotan kelas seimbang (class_weight='balanced') dengan "
            "metrik seleksi utama Macro F1-Score dan Balanced Accuracy."
        ),
    }

    joblib.dump(artefak, PATH_MODEL)
    print(f"Pipeline dan metadata berhasil disimpan ke: {PATH_MODEL}")


if __name__ == "__main__":
    main()