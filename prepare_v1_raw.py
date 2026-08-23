import pandas as pd
import json

df = pd.read_csv("data/iris.csv")

with open("data/iris_v1_raw.jsonl", "w") as f:
    for _, row in df.iterrows():
        input_text = (
            f"sepal_length: {row['sepal_length']}, "
            f"sepal_width: {row['sepal_width']}, "
            f"petal_length: {row['petal_length']}, "
            f"petal_width: {row['petal_width']}"
        )
        record = {"input_text": input_text, "output_text": row["species"]}
        f.write(json.dumps(record) + "\n")

print(f"Wrote {len(df)} records to data/iris_v1_raw.jsonl")
