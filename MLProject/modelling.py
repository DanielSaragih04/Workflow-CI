import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "diabetes_preprocessing",
    "diabetes_preprocessed.csv"
)
EXPERIMENT   = "Diabetes_Prediction_Basic"
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  Modelling — Diabetes Prediction (Basic / Autolog)")
print("  by Daniel Dermawansyah Putra Saragih")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n[1/4] Dataset dimuat: {df.shape}")

X = df.drop(columns=["diabetes"])
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 2. SETUP MLFLOW
# ─────────────────────────────────────────────
# Untuk DagsHub (Advance), ganti bagian ini dengan:
#   import dagshub
#   dagshub.init(repo_owner="username", repo_name="repo", mlflow=True)
#   mlflow.set_tracking_uri("https://dagshub.com/username/repo.mlflow")

# mlflow.set_experiment(EXPERIMENT)
print("\n[2/4] MLflow tracking aktif")

# ─────────────────────────────────────────────
# 3. TRAINING DENGAN AUTOLOG
# ─────────────────────────────────────────────
print("\n[3/4] Melatih model RandomForestClassifier...")

# mlflow.sklearn.autolog()

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Training dimulai...")
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/random_forest_model.pkl"
)

print("Model berhasil disimpan.")

y_pred = model.predict(X_test)

# Tampilkan metrik di terminal
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", auc)

print("\n[4/4] Hasil Evaluasi Model:")
print(f"      Accuracy : {acc:.4f}")
print(f"      Precision: {prec:.4f}")
print(f"      Recall   : {rec:.4f}")
print(f"      F1-Score : {f1:.4f}")
print(f"      ROC-AUC  : {auc:.4f}")

print("\n✅ Training selesai! Buka MLflow UI dengan perintah:")
print("   mlflow ui")
print("   Lalu akses http://127.0.0.1:5000\n")

# Workflow CI Updated