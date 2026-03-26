"""Tests for LSTM, GRU, and Transformer models."""

import torch
import pytest

from src.models import LSTMPredictor, GRUPredictor, TransformerPredictor, build_model


@pytest.fixture
def sample_input():
    """Batch of 4 sequences, 30 timesteps, 12 features."""
    return torch.randn(4, 30, 12)


def test_lstm_output_shape(sample_input):
    model = LSTMPredictor(input_size=12, hidden_size=32, num_layers=2)
    out = model(sample_input)
    assert out.shape == (4,)


def test_transformer_output_shape(sample_input):
    model = TransformerPredictor(input_size=12, d_model=32, nhead=4, num_layers=2)
    out = model(sample_input)
    assert out.shape == (4,)


def test_build_model_lstm():
    model = build_model("lstm", input_size=10, hidden_size=16)
    assert isinstance(model, LSTMPredictor)


def test_build_model_transformer():
    model = build_model("transformer", input_size=10, d_model=32, nhead=4)
    assert isinstance(model, TransformerPredictor)


def test_gru_output_shape(sample_input):
    model = GRUPredictor(input_size=12, hidden_size=32, num_layers=2)
    out = model(sample_input)
    assert out.shape == (4,)


def test_build_model_gru():
    model = build_model("gru", input_size=10, hidden_size=16)
    assert isinstance(model, GRUPredictor)


def test_build_model_invalid():
    with pytest.raises(ValueError):
        build_model("invalid", input_size=10)


def test_models_trainable(sample_input):
    """Verify all models can do a forward-backward pass."""
    targets = torch.randn(4)
    criterion = torch.nn.MSELoss()

    for model_type in ["lstm", "gru", "transformer"]:
        model = build_model(model_type, input_size=12)
        out = model(sample_input)
        loss = criterion(out, targets)
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


def test_gru_classification(sample_input):
    """GRU classification mode should output values in [0, 1]."""
    model = GRUPredictor(input_size=12, hidden_size=32, classification=True)
    out = model(sample_input)
    assert (out >= 0).all() and (out <= 1).all()
