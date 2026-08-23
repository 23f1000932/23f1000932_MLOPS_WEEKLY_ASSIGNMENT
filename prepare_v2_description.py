import pandas as pd
import json

df = pd.read_csv("data/iris.csv")

with open("data/iris_v2_description.jsonl", "w") as f:
    for _, row in df.iterrows():
        input_text = (
            f"A flower specimen has a sepal length of {row['sepal_length']} cm, "
            f"sepal width of {row['sepal_width']} cm, "
            f"petal length of {row['petal_length']} cm, "
            f"and petal width of {row['petal_width']} cm. "
            f"Identify the iris species."
        )
        output_text = f"This is Iris {row['species']}."
        record = {"input_text": input_text, "output_text": output_text}
        f.write(json.dumps(record) + "\n")

print(f"Wrote {len(df)} records to data/iris_v2_description.jsonl")
