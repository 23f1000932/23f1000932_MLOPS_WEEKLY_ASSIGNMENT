import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

RANDOM_SEED = 42
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

original = pd.read_csv("data/iris.csv")

# Simulate a production dataset: shift petal_length up by 1.5 (e.g. a new
# growing region/season), leave the other three features untouched.
rng = np.random.default_rng(RANDOM_SEED)
production = original.copy()
production["petal_length"] = production["petal_length"] + 1.5
production.to_csv("data/iris_production_simulated.csv", index=False)

print("Data drift check (Kolmogorov-Smirnov test per feature):\n")
results = []
for feature in FEATURES:
    stat, p_value = stats.ks_2samp(original[feature], production[feature])
    drifted = p_value < 0.05
    results.append({"feature": feature, "ks_statistic": stat, "p_value": p_value, "drifted": drifted})
    print(f"{feature:15s} KS stat={stat:.4f}  p-value={p_value:.6f}  drifted={drifted}")

results_df = pd.DataFrame(results)
results_df.to_csv("drift_results.csv", index=False)

# Plot original vs production distribution for each feature
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feature in zip(axes.flat, FEATURES):
    ax.hist(original[feature], bins=15, alpha=0.5, label="original", color="tab:blue")
    ax.hist(production[feature], bins=15, alpha=0.5, label="production (simulated)", color="tab:red")
    ax.set_title(feature)
    ax.legend()
plt.tight_layout()
plt.savefig("drift_distributions.png", dpi=150)
plt.close()
print("\nSaved drift_distributions.png and drift_results.csv")
