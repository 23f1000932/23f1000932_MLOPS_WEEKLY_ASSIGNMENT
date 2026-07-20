import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Task 2: Point MLflow to a local tracking directory and name the experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_classification")

# Load data
df = pd.read_csv("data/iris.csv")
train, eval_set = train_test_split(
    df, test_size=0.2, stratify=df["species"], random_state=42
)

X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train["species"]
X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_eval = eval_set["species"]

# Task 1: Hyperparameter configurations to try
hyperparameter_configs = [
    {"max_depth": 2, "min_samples_split": 2},
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 4, "min_samples_split": 4},
    {"max_depth": 5, "min_samples_split": 6},
]

for config in hyperparameter_configs:
    with mlflow.start_run():
        model = DecisionTreeClassifier(
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_eval)
        accuracy = metrics.accuracy_score(y_eval, predictions)
        precision = metrics.precision_score(y_eval, predictions, average="macro")
        recall = metrics.recall_score(y_eval, predictions, average="macro")

        # Log hyperparameters
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("min_samples_split", config["min_samples_split"])

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged run - Config: {config} | Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")