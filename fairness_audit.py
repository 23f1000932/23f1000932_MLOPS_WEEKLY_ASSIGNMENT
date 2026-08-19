import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from fairlearn.metrics import MetricFrame

df = pd.read_csv("data/iris_with_location.csv")

train, eval_set = train_test_split(
    df, test_size=0.2, stratify=df["species"], random_state=42
)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train = train[FEATURES]
y_train = train["species"]
X_eval = eval_set[FEATURES]
y_eval = eval_set["species"]
location_eval = eval_set["location"]  # sensitive attribute, NOT a training feature

model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_eval)

metric_funcs = {
    "accuracy": metrics.accuracy_score,
    "precision": lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall": lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average="macro", zero_division=0),
}

mf = MetricFrame(
    metrics=metric_funcs,
    y_true=y_eval,
    y_pred=predictions,
    sensitive_features=location_eval,
)

print("Overall metrics:")
print(mf.overall)
print("\nMetrics by location group:")
print(mf.by_group)
print("\nDifference between groups (max - min):")
print(mf.difference())
