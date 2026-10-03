import time
import statistics
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data import get_splits
from src.train import BEST_PARAMS, run_cv

MIN_F1 = 0.88
MAX_LATENCY_MS = 30


@pytest.fixture(scope="module")
def model_and_data():
    X_train, X_test, y_train, y_test = get_splits()
    model = RandomForestClassifier(**BEST_PARAMS, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def test_metric_gate():
    """Gate protects against degraded model performance."""
    X_train, _, y_train, _ = get_splits()
    model = RandomForestClassifier(**BEST_PARAMS, random_state=42, n_jobs=1)
    cv_metrics = run_cv(model, X_train, y_train)
    assert cv_metrics["val_f1_macro"] >= MIN_F1


def test_latency_gate(model_and_data):
    """Gate protects against models that are too slow for inference."""
    model, _, X_test, _, _ = model_and_data

    # Warm up
    model.predict(X_test)

    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(X_test)
        latencies.append((time.perf_counter() - start) * 1000)

    median_latency = statistics.median(latencies)
    assert median_latency <= MAX_LATENCY_MS


def test_schema_gate(model_and_data):
    """Gate protects against unexpected output types or shapes."""
    model, _, X_test, _, _ = model_and_data
    predictions = model.predict(X_test)

    assert np.issubdtype(predictions.dtype, np.integer)
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1, 2})
