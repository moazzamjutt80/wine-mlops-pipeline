import numpy as np
import pandas as pd
import pytest
from src.data import get_splits, load_data, validate_data


def test_split_sizes():
    """Test if split sizes are 142 train and 36 test."""
    X_train, X_test, y_train, y_test = get_splits()
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_train) == 142
    assert len(y_test) == 36


def test_no_nulls():
    """Test that the loaded dataset has no null values."""
    X, y = load_data()
    assert not X.isnull().values.any()
    assert not y.isnull().values.any()


def test_13_features():
    """Test that the loaded dataset has exactly 13 features."""
    X, _ = load_data()
    assert X.shape[1] == 13


def test_stratification():
    """Test if class proportions in train and test are close to the full dataset."""
    X, y = load_data()
    _, _, y_train, y_test = get_splits()

    prop_full = y.value_counts(normalize=True).sort_index()
    prop_train = y_train.value_counts(normalize=True).sort_index()
    prop_test = y_test.value_counts(normalize=True).sort_index()

    pd.testing.assert_series_equal(prop_train, prop_full, check_exact=False, atol=0.05)
    pd.testing.assert_series_equal(prop_test, prop_full, check_exact=False, atol=0.05)


def test_reproducibility():
    """Test if two calls to get_splits give identical splits."""
    X_train1, X_test1, y_train1, y_test1 = get_splits()
    X_train2, X_test2, y_train2, y_test2 = get_splits()

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_validate_data_raises_error():
    """Test that validate_data raises an error when given bad data."""
    X, y = load_data()

    # Test null values in X
    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Data contains null values."):
        validate_data(X_bad, y)

    # Test wrong feature count
    X_bad_features = X.drop(X.columns[0], axis=1)
    with pytest.raises(ValueError, match="Expected 13 features"):
        validate_data(X_bad_features, y)
