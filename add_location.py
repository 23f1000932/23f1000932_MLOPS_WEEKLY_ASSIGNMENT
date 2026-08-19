import pandas as pd
import numpy as np

RANDOM_SEED = 42

def add_location_column(input_path="data/iris.csv", output_path="data/iris_with_location.csv"):
    df = pd.read_csv(input_path)
    rng = np.random.default_rng(RANDOM_SEED)
    df["location"] = rng.integers(0, 2, size=len(df))  # 0 or 1, random group assignment
    df.to_csv(output_path, index=False)
    print(f"Added 'location' column -> {output_path}")
    print(df["location"].value_counts())
    return df

if __name__ == "__main__":
    add_location_column()
