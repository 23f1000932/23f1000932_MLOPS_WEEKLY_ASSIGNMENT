import pandas as pd
import numpy as np

df = pd.read_csv("data/iris.csv")
np.random.seed(1)
extra = df.sample(30, replace=True, random_state=1).copy()
for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
    extra[col] += np.random.normal(0, 0.1, size=len(extra))

augmented = pd.concat([df, extra], ignore_index=True)
augmented.to_csv("data/iris.csv", index=False)
print("New row count:", len(augmented))

#This script pretends 30 new (slightly noisy) flower measurements just arrived, and adds them to your dataset.