import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import os 
from dotenv import load_dotenv

# --- KONFIGURASI PATHS ---
# Sesuai permintaan Anda: HANYA NAMA FILE
CLEAN_DATA_PATH = 'healthcare-dataset-stroke-data_preprocessing.csv' 
EXPERIMENT_NAME = "CI_XGBoost_Retraining"
MODEL_NAME = "XGBoostStrokePredictor_CI" 


def load_data(filename):
    """Memuat data bersih dengan pencarian lokasi otomatis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Daftar lokasi kemungkinan file berada
    possible_paths = [
        os.path.join(script_dir, filename),                     # 1. Di folder yang sama (MLProject)
        os.path.join(script_dir, '..', '..', 'preprocessing', filename), # 2. Di folder preprocessing (dari Workflow-CI/MLProject)
        os.path.join(script_dir, '..', 'preprocessing', filename)        # 3. Alternatif path relatif
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    if found_path:
        print(f"✅ Data ditemukan di: {found_path}")
        try:
            df = pd.read_csv(found_path)
            X = df.drop('stroke', axis=1)
            y = df['stroke']
            return X, y
        except Exception as e:
            print(f"Error saat membaca data: {e}")
            exit(1)
    else:
        print(f"❌ Error: File '{filename}' tidak ditemukan di lokasi mana pun:")
        for p in possible_paths:
            print(f" - {p}")
        exit(1)

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Melatih model XGBoost dasar."""
    
    # --- 1. Persiapan MLflow ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Matikan autolog agar tidak konflik dengan manual logging
    mlflow.xgboost.autolog(disable=True) 

    # --- 2. RESAMPLING dengan SMOTE ---
    print("Mulai proses Resampling dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # --- 3. Pelatihan Model XGBoost Baseline ---
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42,
        n_estimators=100 
    )
    
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    # Menghitung metrik manual untuk display
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Baseline Selesai Dilatih ---")
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score (XGBoost): {auc:.4f}")
    
    # Logging metrik tambahan secara manual
    mlflow.log_metric("AUC_ROC", auc)
    mlflow.log_param("smote_applied", True)
    
    # >>> PERBAIKAN UTAMA: SIMPAN MODEL SECARA MANUAL <<<
    # Ini menjamin artifact 'model' ada, sehingga register_model tidak akan error
    print("💾 Menyimpan model ke MLflow Artifacts...")
    mlflow.xgboost.log_model(xgb_model, "model")

    # --- 4. REGISTER MODEL KE DAGSHUB ---
    try:
         run_id = mlflow.active_run().info.run_id
         model_uri = f"runs:/{run_id}/model"
         
         print(f"Mendaftarkan model dari URI: {model_uri}")
         mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
         
         print(f"✅ Model '{MODEL_NAME}' berhasil didaftarkan ke DagsHub.")
    except Exception as e:
        print(f"❌ Gagal mendaftarkan model: {e}")
        # Exit code 1 agar CI gagal jika registrasi gagal
        exit(1)
    
    print("Model dan metrik dicatat.")
    
    # --- MENAMPILKAN HASIL EVALUASI LENGKAP ---
    print("\n--- Confusion Matrix (XGBoost Baseline) ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report (XGBoost Baseline) ---")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split 
    
    # >>> PERBAIKAN LOGIKA .ENV: FORCE OVERRIDE <<<
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_env_path = os.path.join(script_dir, '..', '..', '.env')
    
    print(f"🔍 Mencari file .env di: {os.path.abspath(root_env_path)}")
    
    if os.path.exists(root_env_path):
        # PERBAIKAN: override=True AKAN MEMAKSA MEMUAT ENV DARI FILE .ENV
        load_dotenv(root_env_path, override=True) 
        print("✅ File .env ditemukan dan dimuat (Override Mode Aktif).")
    else:
        print("⚠️ File .env tidak ditemukan. Mengandalkan Environment Variables (CI/CD).")
        load_dotenv() 
    
    # CEK URI DAN PRINT STATUS
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    
    if tracking_uri:
        print(f"🔗 TARGET CONNECT: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        
        if "dagshub" not in tracking_uri:
             print("⚠️ PERINGATAN: URI masih terlihat lokal! Pastikan file .env benar.")
    else:
        print("❌ CRITICAL ERROR: MLFLOW_TRACKING_URI tidak ditemukan!")

    # Load data dan split
    X, y = load_data(CLEAN_DATA_PATH)
    
    # Train-Test Split (stratify=y sangat penting untuk data imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Mulai proses training
    train_baseline_model(X_train, y_train, X_test, y_test)