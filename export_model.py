import mlflow
import mlflow.sklearn
import joblib
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "iris_classifier"
MODEL_VERSION = "1"

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

os.makedirs("model_export", exist_ok=True)
joblib.dump(model, "model_export/model.joblib")

print(f"Exported {MODEL_NAME} version {MODEL_VERSION} to model_export/model.joblib")