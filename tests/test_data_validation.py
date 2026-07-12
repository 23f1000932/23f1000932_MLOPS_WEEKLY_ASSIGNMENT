import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "iris.csv")

EXPECTED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
EXPECTED_SPECIES = {"setosa", "versicolor", "virginica"}


def load_data():
    return pd.read_csv(DATA_PATH)


def test_expected_columns_present():
    df = load_data()
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Missing expected column: {col}"


def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"


def test_feature_types_are_numeric():
    df = load_data()
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


def test_species_values_are_valid():
    df = load_data()
    assert set(df["species"].unique()).issubset(EXPECTED_SPECIES), "Unexpected species labels found"


def test_feature_value_ranges_are_reasonable():
    df = load_data()
    assert df["sepal_length"].between(0, 15).all(), "sepal_length out of reasonable range"
    assert df["sepal_width"].between(0, 15).all(), "sepal_width out of reasonable range"
    assert df["petal_length"].between(0, 15).all(), "petal_length out of reasonable range"
    assert df["petal_width"].between(0, 15).all(), "petal_width out of reasonable range"


def test_dataset_not_empty():
    df = load_data()
    assert len(df) > 0, "Dataset is empty"