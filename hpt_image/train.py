import json
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import boto3
import io

print('👋 Iniciando entrenamiento...')

# Leer hiperparámetros desde el JSON de SageMaker
hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
with open(hyperparams_path, "r") as f:
    params = json.load(f)

# Convertir a tipos esperados
year = int(params["year"])
month = int(params["month"])
n_estimators = int(params["n_estimators"])
max_depth = int(params["max_depth"])

print("🧾 Hiperparámetros:")
print(params)

#----------------------------------------------------------------------
# ✅ Lectura del dataset desde S3

print("📥 Leyendo datasets desde S3...")

bucket = "proyecto-1-ml"
prefix = f"preprocessing/{year}_{month:02d}"

s3 = boto3.client("s3")

def read_csv_from_s3(bucket, key):
    print(f"🔸 Leyendo s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

# Parametrizado por año y mes
X_train_path = f"{prefix}/X_train.csv"
X_train = read_csv_from_s3(bucket, X_train_path)
y_train_path = f"{prefix}/y_train.csv"
y_train = read_csv_from_s3(bucket, y_train_path)

X_val_path = f"{prefix}/X_val.csv"
X_val = read_csv_from_s3(bucket, X_val_path)
y_val_path = f"{prefix}/y_val.csv"
y_val = read_csv_from_s3(bucket, y_val_path)

#----------------------------------------------------------------------

# Entrenar modelo
print("Entrenando modelo...")
model = xgb.XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)

model = model.fit(X_train, y_train)

# Train

y_train_pred_prob = model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_pred_prob >= 0.25)

f1_train = f1_score(y_train, y_train_pred, average = 'macro')
print('f1 train:', round(f1_train*100, 2))


# Val

y_val_pred_prob = model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_pred_prob >= 0.25)

f1 = f1_score(y_val, y_val_pred, average = 'macro')
print('f1_score:', round(f1*100, 2))

#----------------------------------------------------------------------

print("💾 Guardando modelo en /opt/ml/model/model.joblib...")
os.makedirs("/opt/ml/model", exist_ok=True)
model.save_model("/opt/ml/model/model.json")
print("✅ Modelo guardado correctamente.")

print("✅ Versión de XGBoost:", xgb.__version__)

#----------------------------------------------------------------------

print("✅ Entrenamiento y evaluación completados.")
