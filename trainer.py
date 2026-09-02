"""Training / evaluation loop for SELM.

Two solvers are supported (config `train.solver`):
  * "closed_form": one-shot ridge-regression fit of the output layer
    (Eq. 8 of the paper) -- no epochs, no optimizer.
  * "sgd": mini-batch gradient descent on the (regularized) MSE loss,
    matching Algorithm 1 of the paper.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SELM
from utils import mae, rmse, save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: SELM, device: torch.device, train_cfg: dict[str, Any], checkpoint_dir: str | Path):
        self.model = model.to(device)
        self.device = device
        self.cfg = train_cfg
        self.checkpoint_dir = Path(checkpoint_dir)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg["optimizer"] == "adam":
            return torch.optim.Adam(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        if self.cfg["optimizer"] == "sgd":
            return torch.optim.SGD(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        raise ValueError(f"Unknown optimizer '{self.cfg['optimizer']}'")

    def _to_device(self, x_num, x_cat, y=None):
        x_num = x_num.to(self.device)
        x_cat = {k: v.to(self.device) for k, v in x_cat.items()}
        if y is not None:
            return x_num, x_cat, y.to(self.device)
        return x_num, x_cat

    def fit_closed_form(self, train_loader: DataLoader) -> None:
        logger.info("Fitting SELM output layer in closed form (ridge regression, Eq. 8)...")
        self.model.freeze_embeddings()
        self.model.closed_form_fit(train_loader, ridge_lambda=self.cfg["ridge_lambda"], device=self.device)
        save_checkpoint(self.model, self.checkpoint_dir / "selm_closed_form.pt")

    def fit_sgd(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        optimizer = self._make_optimizer()
        loss_fn = nn.MSELoss()

        best_val_mae = float("inf")
        best_state = None
        patience = self.cfg["early_stopping_patience"]
        epochs_without_improvement = 0

        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            running_loss = 0.0
            for step, (x_num, x_cat, y) in enumerate(train_loader, start=1):
                x_num, x_cat, y = self._to_device(x_num, x_cat, y)

                optimizer.zero_grad()
                preds = self.model(x_num, x_cat)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if step % self.cfg["log_every"] == 0:
                    logger.info("epoch %d step %d: train MSE=%.4f", epoch, step, running_loss / step)

            val_mae, val_rmse = self.evaluate(val_loader)
            logger.info(
                "epoch %d done: train MSE=%.4f val MAE=%.4f val RMSE=%.4f",
                epoch,
                running_loss / max(1, len(train_loader)),
                val_mae,
                val_rmse,
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping at epoch %d (best val MAE=%.4f)", epoch, best_val_mae)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        save_checkpoint(self.model, self.checkpoint_dir / "selm_sgd.pt")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        all_preds, all_targets = [], []
        for x_num, x_cat, y in loader:
            x_num, x_cat, y = self._to_device(x_num, x_cat, y)
            preds = self.model(x_num, x_cat)
            all_preds.append(preds)
            all_targets.append(y)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        return mae(preds, targets), rmse(preds, targets)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        for batch in loader:
            x_num, x_cat = batch[0], batch[1]
            x_num, x_cat = self._to_device(x_num, x_cat)
            all_preds.append(self.model(x_num, x_cat).cpu())
        return torch.cat(all_preds).squeeze(1).numpy()
