"""LSTM and Transformer models for price prediction.

Cycle 5: Added classification mode (direction prediction with sigmoid output).
"""

import math

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """LSTM model for time-series prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        classification: bool = False,
    ):
        super().__init__()
        self.classification = classification
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(-1)
        if self.classification:
            return torch.sigmoid(logits)
        return logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerPredictor(nn.Module):
    """Transformer encoder model for time-series prediction."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        classification: bool = False,
    ):
        super().__init__()
        self.classification = classification
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        last = x[:, -1, :]  # take last timestep
        out = self.dropout(last)
        logits = self.fc(out).squeeze(-1)
        if self.classification:
            return torch.sigmoid(logits)
        return logits


def build_model(model_type: str, input_size: int, **kwargs) -> nn.Module:
    """Factory function to build a model."""
    if model_type == "lstm":
        valid = {"hidden_size", "num_layers", "dropout", "classification"}
        return LSTMPredictor(input_size, **{k: v for k, v in kwargs.items() if k in valid})
    elif model_type == "transformer":
        valid = {"d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "classification"}
        return TransformerPredictor(input_size, **{k: v for k, v in kwargs.items() if k in valid})
    else:
        raise ValueError(f"Unknown model type: {model_type}")
