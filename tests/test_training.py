"""Tests for training and data preparation."""

import numpy as np
import pytest

from src.training import TimeSeriesDataset, prepare_data, train_model, predict
from src.models import build_model


def test_time_series_dataset():
    features = np.random.randn(100, 5)
    targets = np.random.randn(100)
    ds = TimeSeriesDataset(features, targets, seq_len=10)
    assert len(ds) == 90
    x, y = ds[0]
    assert x.shape == (10, 5)
    assert y.shape == ()


def test_prepare_data_no_leakage():
    """Ensure scaling stats come from train only."""
    np.random.seed(42)
    features = np.random.randn(200, 5)
    # Make test data have very different distribution
    features[150:] += 100
    targets = np.random.randn(200)

    train_ds, test_ds, stats, offset = prepare_data(features, targets, train_end=150, seq_len=10)

    # Stats should be from train data only (mean near 0, not near 50)
    assert np.all(np.abs(stats["mean"]) < 1.0)
    assert len(train_ds) == 140  # 150 - 10
    assert len(test_ds) == 40   # 50 - 10


def test_train_and_predict():
    """End-to-end: train a small model and predict."""
    np.random.seed(42)
    features = np.random.randn(200, 5)
    targets = np.random.randn(200)

    train_ds, test_ds, _, _ = prepare_data(features, targets, train_end=150, seq_len=10)
    model = build_model("lstm", input_size=5, hidden_size=16, num_layers=1)

    losses = train_model(model, train_ds, epochs=5, batch_size=16)
    assert len(losses) == 5
    assert all(l > 0 for l in losses)

    preds = predict(model, test_ds)
    assert len(preds) == len(test_ds)


def test_train_with_lr_scheduler():
    """Cycle 4: Verify training works with LR scheduler enabled."""
    np.random.seed(42)
    features = np.random.randn(200, 5)
    targets = np.random.randn(200)

    train_ds, _, _, _ = prepare_data(features, targets, train_end=150, seq_len=10)
    model = build_model("lstm", input_size=5, hidden_size=16, num_layers=1)

    losses = train_model(model, train_ds, epochs=10, batch_size=16, use_lr_scheduler=True)
    assert len(losses) == 10
    assert all(l > 0 for l in losses)


def test_train_without_lr_scheduler():
    """Cycle 4: Verify training works with LR scheduler disabled."""
    np.random.seed(42)
    features = np.random.randn(200, 5)
    targets = np.random.randn(200)

    train_ds, _, _, _ = prepare_data(features, targets, train_end=150, seq_len=10)
    model = build_model("lstm", input_size=5, hidden_size=16, num_layers=1)

    losses = train_model(model, train_ds, epochs=5, batch_size=16, use_lr_scheduler=False)
    assert len(losses) == 5
