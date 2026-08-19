import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris_with_location.csv")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

train, eval_set = train_test_split(
    df, test_size=0.2, stratify=df["species"], random_state=42
)
X_train = train[FEATURES]
y_train = train["species"]
X_eval = eval_set[FEATURES]

model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_eval)

# model.classes_ gives the class order matching shap_values' last dimension
classes = model.classes_
print("Class order:", list(classes))

for i, class_name in enumerate(classes):
    plt.figure()
    shap.summary_plot(shap_values[:, :, i], X_eval, show=False)
    plt.title(f"SHAP Summary — {class_name}")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{class_name}.png", dpi=150)
    plt.close()
    print(f"Saved shap_summary_{class_name}.png")
