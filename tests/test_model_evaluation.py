import os
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "iris.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.joblib")

MIN_ACCURACY = 0.85


def load_model_and_eval_set():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    _, eval_set = train_test_split(
        df, test_size=0.2, stratify=df["species"], random_state=42
    )
    X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_eval = eval_set["species"]
    return model, X_eval, y_eval

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Trained model file not found"


def test_model_accuracy_meets_threshold():
    model, X_eval, y_eval = load_model_and_eval_set()
    predictions = model.predict(X_eval)
    accuracy = metrics.accuracy_score(y_eval, predictions)
    assert accuracy >= MIN_ACCURACY, f"Model accuracy {accuracy:.3f} is below threshold {MIN_ACCURACY}"


def test_model_predicts_valid_labels():
    model, X_eval, y_eval = load_model_and_eval_set()
    predictions = model.predict(X_eval)
    valid_labels = {"setosa", "versicolor", "virginica"}
    assert set(predictions).issubset(valid_labels), "Model predicted an invalid species label"