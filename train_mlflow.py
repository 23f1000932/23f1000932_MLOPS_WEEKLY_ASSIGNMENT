import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
# We vary two hyperparameters: max_depth and min_samples_split
hyperparameter_configs = [
    {"max_depth": 2, "min_samples_split": 2},
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 4, "min_samples_split": 4},
    {"max_depth": 5, "min_samples_split": 6},
]

for config in hyperparameter_configs:
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

    print(f"Config: {config} | Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")