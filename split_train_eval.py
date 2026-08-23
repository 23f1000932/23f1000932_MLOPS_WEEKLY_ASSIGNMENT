import pandas as pd
import json
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.csv")

# Same split logic used every week: 80/20, stratified, same seed
train_idx, eval_idx = train_test_split(
    df.index, test_size=0.2, stratify=df["species"], random_state=42
)

def split_jsonl(input_path, train_path, eval_path):
    with open(input_path) as f:
        lines = f.readlines()
    with open(train_path, "w") as f:
        f.writelines(lines[i] for i in train_idx)
    with open(eval_path, "w") as f:
        f.writelines(lines[i] for i in eval_idx)
    print(f"{input_path} -> {len(train_idx)} train, {len(eval_idx)} eval")

split_jsonl("data/iris_v1_raw.jsonl", "data/iris_v1_train.jsonl", "data/iris_v1_eval.jsonl")
split_jsonl("data/iris_v2_description.jsonl", "data/iris_v2_train.jsonl", "data/iris_v2_eval.jsonl")
