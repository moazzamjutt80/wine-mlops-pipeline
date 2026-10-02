import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the Wine dataset as a pandas DataFrame X and Series y."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Validates the data to ensure no null values and exactly 13 features."""
    if X.isnull().values.any() or y.isnull().values.any():
        raise ValueError("Data contains null values.")
    if X.shape[1] != 13:
        raise ValueError(f"Expected 13 features, got {X.shape[1]}")


def get_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads, validates, and performs a stratified 80/20 train-test split."""
    X, y = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
