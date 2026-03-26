"""Training and dataset utilities.

Cycle 4: Added learning rate scheduling (ReduceLROnPlateau).
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time-series prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        self.features = features
        self.targets = targets
        self.n_samples = len(features) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_data(
    features: np.ndarray,
    targets: np.ndarray,
    train_end: int,
    seq_len: int,
    scaler_stats: dict | None = None,
) -> tuple:
    """Split and scale data using train-only statistics.

    Returns (train_dataset, test_dataset, scaler_stats, test_indices_offset).
    test_indices_offset is the index in the original array where test predictions start.
    """
    train_features = features[:train_end]
    train_targets = targets[:train_end]
    test_features = features[train_end:]
    test_targets = targets[train_end:]

    # Compute scaling stats from train only
    if scaler_stats is None:
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid division by zero
        scaler_stats = {"mean": mean, "std": std}

    # Scale both with train stats
    train_scaled = (train_features - scaler_stats["mean"]) / scaler_stats["std"]
    test_scaled = (test_features - scaler_stats["mean"]) / scaler_stats["std"]

    train_ds = TimeSeriesDataset(train_scaled, train_targets, seq_len)
    test_ds = TimeSeriesDataset(test_scaled, test_targets, seq_len)

    return train_ds, test_ds, scaler_stats, train_end + seq_len


def train_model(
    model: torch.nn.Module,
    train_ds: Dataset,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    use_lr_scheduler: bool = True,
) -> list[float]:
    """Train model and return loss history.

    Cycle 4: Added ReduceLROnPlateau scheduler for better convergence.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if scheduler is not None:
            scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.2e}")

    return losses


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cpu",
) -> np.ndarray:
    """Generate predictions from a dataset."""
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds = []
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])
