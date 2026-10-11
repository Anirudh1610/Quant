"""train_lstm.py — PyTorch LSTM for NIFTY volatility forecasting.

Architecture (mirrors the requested Keras spec):
    LSTM(64, return_sequences=True) → Dropout(0.2)
    LSTM(32)                        → Dropout(0.2)
    Linear(32→16, ReLU)
    Linear(16→1)

Note: TensorFlow/Keras does not support Python 3.14 on arm64 as of mid-2025.
PyTorch 2.x is used as a drop-in replacement with identical semantics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import get_logger, load_config, set_seed, ensure_dirs, project_root

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityLSTM(nn.Module):
    """
    Stacked LSTM for volatility regression.

    Parameters
    ----------
    input_size : number of input features per timestep
    hidden1    : units in first LSTM layer
    hidden2    : units in second LSTM layer
    fc_hidden  : units in the dense hidden layer
    dropout    : dropout probability applied after each LSTM layer
    """

    def __init__(
        self,
        input_size: int,
        hidden1: int = 64,
        hidden2: int = 32,
        fc_hidden: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm1(x)          # (batch, seq_len, hidden1)
        out = self.drop1(out)
        out, _ = self.lstm2(out)        # (batch, seq_len, hidden2)
        out = self.drop2(out)
        out = out[:, -1, :]             # take last timestep → (batch, hidden2)
        out = self.relu(self.fc1(out))  # (batch, fc_hidden)
        return self.fc2(out).squeeze(-1)  # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights: dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class LSTMTrainer:
    """Encapsulates the full train / evaluate lifecycle."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.mc = cfg["model"]
        self.device = get_device()
        self.model: VolatilityLSTM | None = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        model_dir = project_root() / self.mc["model_dir"]
        ensure_dirs(model_dir)
        self.ckpt_path = model_dir / self.mc["model_name"]

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, input_size: int) -> "LSTMTrainer":
        set_seed(self.mc["seed"])
        self.model = VolatilityLSTM(
            input_size=input_size,
            hidden1=self.mc["hidden1"],
            hidden2=self.mc["hidden2"],
            fc_hidden=self.mc["fc_hidden"],
            dropout=self.mc["dropout"],
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Model: %s  |  device: %s  |  params: %d",
            self.model.__class__.__name__, self.device, n_params,
        )
        return self

    # ── train ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "LSTMTrainer":
        assert self.model is not None, "Call build() first"

        batch_size = self.mc["batch_size"]
        epochs = self.mc["epochs"]
        lr = self.mc["learning_rate"]

        train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.mc["lr_patience"],
            factor=self.mc["lr_factor"],
            min_lr=self.mc["min_lr"],
        )
        early_stop = EarlyStopping(patience=self.mc["patience"])

        logger.info("Training for up to %d epochs …", epochs)
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            # ── train step ────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(train_loader.dataset)

            # ── val step ──────────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    pred = self.model(Xb)
                    val_loss += criterion(pred, yb).item() * len(Xb)
            val_loss /= len(val_loader.dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.2e  [%.0fs]",
                    epoch, epochs, train_loss, val_loss, lr_now, elapsed,
                )

            if early_stop(val_loss, self.model):
                logger.info("Early stopping at epoch %d (best val=%.6f)", epoch, early_stop.best_loss)
                early_stop.restore_best(self.model)
                break

        # Save best checkpoint
        self.save()
        logger.info("Training complete in %.0fs", time.time() - t0)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        assert self.model is not None, "Model not built/loaded"
        bs = batch_size or self.mc["batch_size"]
        self.model.eval()
        all_preds = []
        loader = _make_loader(X, np.zeros(len(X)), bs, shuffle=False)
        for Xb, _ in loader:
            Xb = Xb.to(self.device)
            all_preds.append(self.model(Xb).cpu().numpy())
        return np.concatenate(all_preds)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        assert self.model is not None
        p = Path(path) if path else self.ckpt_path
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "history": self.history,
                "cfg": self.mc,
            },
            p,
        )
        logger.info("Model checkpoint saved → %s", p)
        return p

    def load(self, path: str | Path | None = None, input_size: int | None = None) -> "LSTMTrainer":
        p = Path(path) if path else self.ckpt_path
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        saved_cfg = ckpt.get("cfg", self.mc)
        n_in = input_size or saved_cfg.get("input_size", self.mc.get("input_size", 1))
        self.model = VolatilityLSTM(
            input_size=n_in,
            hidden1=saved_cfg["hidden1"],
            hidden2=saved_cfg["hidden2"],
            fc_hidden=saved_cfg["fc_hidden"],
            dropout=saved_cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", {})
        logger.info("Model loaded ← %s", p)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build + train in one call
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> LSTMTrainer:
    """Full pipeline: build model, train, return fitted trainer."""
    trainer = LSTMTrainer(cfg)
    trainer.build(input_size=X_train.shape[2])
    trainer.fit(X_train, y_train, X_val, y_val)
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward training loop
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_train(
    df: pd.DataFrame,
    windows: list[dict],
    cfg: dict[str, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> list[dict]:
    """
    Retrain the LSTM from scratch on each walk-forward window.

    Returns
    -------
    List of dicts: {test_start, test_end, dates, predictions, actuals}
    """
    from dataset_builder import VolatilityDataset
    from feature_engineering import FEATURE_COLS, TARGET_COL

    if cfg is None:
        cfg = load_config()
    feature_cols = feature_cols or FEATURE_COLS
    results = []

    for i, win in enumerate(windows):
        train_end = win["train_end"]
        test_start = win["test_start"]
        test_end = win["test_end"]
        logger.info(
            "Walk-forward window %d/%d: train→%s  test %s→%s",
            i + 1, len(windows), train_end.date(), test_start.date(), test_end.date(),
        )

        train_df = df[df.index <= train_end]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]

        if len(train_df) < cfg["features"]["lookback"] + 30:
            logger.warning("Not enough training data in window %d — skipping", i + 1)
            continue

        # Build mini-dataset using train window only for scaling
        temp_cfg = dict(cfg)
        temp_cfg["dates"] = dict(cfg["dates"])
        temp_cfg["dates"]["train_end"] = str(train_end.date())
        temp_cfg["dates"]["val_end"] = str(train_end.date())

        ds = VolatilityDataset(train_df, cfg=temp_cfg, feature_cols=feature_cols)
        ds.fit_and_transform()

        if ds.X_train is None or len(ds.X_train) == 0:
            continue

        trainer = train_model(ds.X_train, ds.y_train, ds.X_val, ds.y_val, cfg)

        # Predict on test window (need lookback context from end of train)
        lb = cfg["features"]["lookback"]
        context_df = df[df.index <= test_end].tail(lb + len(test_df))
        context_feats = ds.scaler.transform(
            context_df[feature_cols].ffill().bfill().values
        )
        X_test, y_test, test_dates = _make_test_seqs(
            context_feats,
            context_df[TARGET_COL].values,
            context_df.index,
            lb,
            test_start,
        )

        if len(X_test) == 0:
            continue

        preds = trainer.predict(X_test)
        results.append({
            "test_start": test_start,
            "test_end": test_end,
            "dates": test_dates,
            "predictions": preds,
            "actuals": y_test,
        })

    return results


def _make_test_seqs(
    feats: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int,
    test_start: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Extract only the test-period sequences from a context+test array."""
    from dataset_builder import make_sequences

    X, y, seq_dates = make_sequences(feats, targets, dates, lookback)
    mask = seq_dates >= test_start
    return X[mask], y[mask], seq_dates[mask]
