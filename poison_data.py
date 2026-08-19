import pandas as pd
import numpy as np

RANDOM_SEED = 42
SPECIES = ["setosa", "versicolor", "virginica"]

# Observed min/max per feature (from data/iris.csv describe())
FEATURE_RANGES = {
    "sepal_length": (4.3, 7.9),
    "sepal_width": (2.0, 4.4),
    "petal_length": (1.0, 6.9),
    "petal_width": (0.038423, 2.5),
}

def poison_dataset(df, fraction, seed):
    rng = np.random.default_rng(seed)
    poisoned = df.copy().reset_index(drop=True)
    n_corrupt = int(len(poisoned) * fraction)
    corrupt_idx = rng.choice(poisoned.index, size=n_corrupt, replace=False)

    for col, (low, high) in FEATURE_RANGES.items():
        poisoned.loc[corrupt_idx, col] = rng.uniform(low, high, size=n_corrupt)

    poisoned.loc[corrupt_idx, "species"] = rng.choice(SPECIES, size=n_corrupt)

    return poisoned, corrupt_idx

if __name__ == "__main__":
    clean = pd.read_csv("data/iris.csv")

    for pct in [5, 10, 50]:
        fraction = pct / 100
        poisoned_df, idx = poison_dataset(clean, fraction, seed=RANDOM_SEED + pct)
        out_path = f"data/iris_poisoned_{pct}.csv"
        poisoned_df.to_csv(out_path, index=False)
        print(f"{pct}% poisoning -> {out_path} | corrupted {len(idx)}/{len(clean)} rows")
