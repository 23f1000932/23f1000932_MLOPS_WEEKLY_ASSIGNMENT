import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Task 5: Load the model from the MLflow Model Registry by name and version
MODEL_NAME = "iris_classifier"
MODEL_VERSION = "1"  # or use "latest" via a different URI format

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

print(f"Loaded model '{MODEL_NAME}' version {MODEL_VERSION} from MLflow Model Registry")

# Load evaluation data
df = pd.read_csv("data/iris.csv")
_, eval_set = train_test_split(
    df, test_size=0.2, stratify=df["species"], random_state=42
)
X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_eval = eval_set["species"]

predictions = model.predict(X_eval)
accuracy = metrics.accuracy_score(y_eval, predictions)
precision = metrics.precision_score(y_eval, predictions, average="macro")
recall = metrics.recall_score(y_eval, predictions, average="macro")

print(f"Evaluation results — Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")