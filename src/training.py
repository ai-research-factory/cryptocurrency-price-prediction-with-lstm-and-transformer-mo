"""Training and dataset utilities.

Cycle 4: Added learning rate scheduling (ReduceLROnPlateau).
Cycle 5: Added classification mode (BCE loss for direction prediction).
Cycle 7: Added linear warm-up LR scheduling for Transformer convergence.
Cycle 8: Added early stopping with validation split to reduce overfitting.
Cycle 9: Fixed early stopping + warmup interaction -- early stopping only
activates after warmup phase completes.
Cycle 11: Added differentiable Sharpe ratio loss for direct risk-adjusted
return optimization, addressing the buy-and-hold approximation problem.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DifferentiableSharpe(nn.Module):
    """Differentiable Sharpe ratio loss for direct optimization.

    Cycle 11: Instead of MSE (which optimizes for prediction accuracy),
    this loss directly optimizes for risk-adjusted returns. The model
    output is treated as a position signal, and the loss is the negative
    Sharpe ratio of the resulting strategy.

    This encourages the model to produce signals that maximize Sharpe
    rather than just minimizing prediction error, addressing the issue
    where models learn to approximate buy-and-hold (OQ#6).
    """

    def __init__(self, ann_factor: float = 252.0, cost_bps: float = 10.0):
        super().__init__()
        self.ann_factor = ann_factor
        self.cost_bps = cost_bps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute negative Sharpe ratio as loss.

        Args:
            predictions: Model output (interpreted as position signal via tanh)
            targets: Actual returns for the next period
        """
        # Convert predictions to soft positions via tanh (range [-1, 1])
        positions = torch.tanh(predictions)

        # Strategy returns = position * actual_return
        strategy_returns = positions * targets

        # Approximate transaction costs from position changes
        # Use soft position changes (no discrete jumps)
        cost = self.cost_bps / 10_000
        if len(positions) > 1:
            position_changes = torch.abs(positions[1:] - positions[:-1])
            costs = position_changes * cost
            # First period has entry cost
            first_cost = torch.abs(positions[0:1]) * cost
            all_costs = torch.cat([first_cost, costs])
            net_returns = strategy_returns - all_costs
        else:
            net_returns = strategy_returns - torch.abs(positions) * cost

        mean_ret = net_returns.mean()
        std_ret = net_returns.std() + 1e-8  # stability

        # Negative Sharpe (we minimize loss, so negate)
        neg_sharpe = -(mean_ret / std_ret) * (self.ann_factor ** 0.5)

        return neg_sharpe


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time-series prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int,
                 classification: bool = False):
        self.seq_len = seq_len
        self.features = features
        self.targets = targets
        self.classification = classification
        self.n_samples = len(features) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        if self.classification:
            y = 1.0 if y > 0 else 0.0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_data(
    features: np.ndarray,
    targets: np.ndarray,
    train_end: int,
    seq_len: int,
    scaler_stats: dict | None = None,
    classification: bool = False,
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

    train_ds = TimeSeriesDataset(train_scaled, train_targets, seq_len, classification=classification)
    test_ds = TimeSeriesDataset(test_scaled, test_targets, seq_len, classification=classification)

    return train_ds, test_ds, scaler_stats, train_end + seq_len


def train_model(
    model: torch.nn.Module,
    train_ds: Dataset,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    use_lr_scheduler: bool = True,
    classification: bool = False,
    warmup_epochs: int = 0,
    early_stopping_patience: int = 0,
    val_fraction: float = 0.1,
    sharpe_loss: bool = False,
    sharpe_loss_weight: float = 0.5,
    ann_factor: float = 252.0,
    cost_bps: float = 10.0,
) -> list[float]:
    """Train model and return loss history.

    Cycle 4: Added ReduceLROnPlateau scheduler for better convergence.
    Cycle 5: Added classification mode with BCE loss.
    Cycle 7: Added linear warm-up for Transformer convergence.
    Cycle 8: Added early stopping with validation split.
    Cycle 11: Added Sharpe-aware loss (blended with MSE) for regression models.
    warmup_epochs: number of epochs for linear LR warm-up from lr/10 to lr.
    early_stopping_patience: stop if val loss doesn't improve for this many epochs (0=disabled).
    val_fraction: fraction of training data to use for validation when early stopping is enabled.
    sharpe_loss: if True, use blended MSE + Sharpe loss for regression (ignored for classification).
    sharpe_loss_weight: weight of Sharpe loss in blend (0=pure MSE, 1=pure Sharpe).
    """
    import copy

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss() if classification else torch.nn.MSELoss()

    # Cycle 11: Sharpe-aware loss (only for regression mode)
    sharpe_criterion = None
    if sharpe_loss and not classification:
        sharpe_criterion = DifferentiableSharpe(
            ann_factor=ann_factor, cost_bps=cost_bps
        ).to(device)

    # Split train_ds into train/val if early stopping is enabled
    val_loader = None
    if early_stopping_patience > 0 and len(train_ds) > 10:
        n_val = max(1, int(len(train_ds) * val_fraction))
        n_train = len(train_ds) - n_val
        # Use last portion as val (preserves temporal ordering)
        train_subset = torch.utils.data.Subset(train_ds, range(n_train))
        val_subset = torch.utils.data.Subset(train_ds, range(n_train, n_train + n_val))
        loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    scheduler = None
    warmup_scheduler = None

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )

    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

    losses = []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            base_loss = criterion(pred, y_batch)
            # Cycle 11: Blend MSE and Sharpe loss for regression
            if sharpe_criterion is not None and len(pred) >= 4:
                s_loss = sharpe_criterion(pred, y_batch)
                loss = (1 - sharpe_loss_weight) * base_loss + sharpe_loss_weight * s_loss
            else:
                loss = base_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        # Warm-up takes priority during warmup_epochs, then ReduceLROnPlateau
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.2e}")

        # Early stopping check (skip during warmup phase to let LR stabilize)
        if val_loader is not None and early_stopping_patience > 0 and epoch >= warmup_epochs:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)
                    val_loss += criterion(pred, y_batch).item()
                    val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)

            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} (val_loss={avg_val_loss:.6f}, "
                            f"best={best_val_loss:.6f})")
                break

    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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
