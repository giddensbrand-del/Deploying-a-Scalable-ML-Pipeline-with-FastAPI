import pytest
import numpy as np
from ml.model import compute_model_metrics, inference, train_model

def test_train_model_returns_model():
    X_train = np.array([
        [25, 40],
        [45, 60],
        [35, 50],
        [23, 30],
    ])
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_inference_returns_predictions():
    X_train = np.array([
        [25, 40],
        [45, 60],
        [35, 50],
        [23, 30],
    ])
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == len(y_train)
    assert set(preds).issubset({0, 1})


def test_compute_model_metrics_returns_values():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1