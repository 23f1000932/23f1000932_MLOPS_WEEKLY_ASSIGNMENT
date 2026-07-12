import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("data/iris.csv")
train, eval_set = train_test_split(
    data, test_size=0.2, stratify=data["species"], random_state=42
)

X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train["species"]
X_eval = eval_set[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_eval = eval_set["species"]

model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

acc = metrics.accuracy_score(y_eval, model.predict(X_eval))
print(f"Train samples: {len(train)} | Eval accuracy: {acc:.3f}")

joblib.dump(model, "model.joblib")