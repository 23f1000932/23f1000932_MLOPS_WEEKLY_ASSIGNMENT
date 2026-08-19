import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_poisoning_analysis")

DATASETS = {
    0: "data/iris.csv",
    5: "data/iris_poisoned_5.csv",
    10: "data/iris_poisoned_10.csv",
    50: "data/iris_poisoned_50.csv",
}

for poison_level, path in DATASETS.items():
    df = pd.read_csv(path)
    train, eval_set = train_test_split(
        df, test_size=0.2, stratify=df["species"], random_state=42
    )
    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train["species"]
    X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_eval = eval_set["species"]

    with mlflow.start_run(run_name=f"poison_{poison_level}pct"):
        model = DecisionTreeClassifier(max_depth=3, random_state=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_eval)

        accuracy = metrics.accuracy_score(y_eval, predictions)
        precision = metrics.precision_score(y_eval, predictions, average="macro", zero_division=0)
        recall = metrics.recall_score(y_eval, predictions, average="macro", zero_division=0)
        f1 = metrics.f1_score(y_eval, predictions, average="macro", zero_division=0)

        mlflow.log_param("poisoning_level_pct", poison_level)
        mlflow.log_param("dataset_path", path)
        mlflow.log_param("max_depth", 3)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"Poisoning {poison_level}% -> acc={accuracy:.3f} prec={precision:.3f} rec={recall:.3f} f1={f1:.3f}")
