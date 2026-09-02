"""Shallow Extreme Learning Machine (SELM).

Implements the architecture of Oneto et al. (2018), Section 3.1 / Fig. 4:
a single hidden layer with RANDOM, FROZEN weights W and a nonlinear
activation phi (Eq. 2-3), followed by a trainable, bias-free linear output
layer W* (Eq. 3). Only W* is a learned parameter — that is what makes an
ELM "extreme": no backprop through the hidden layer is needed.

Two ways to fit W* are provided, mirroring the paper:
  * `closed_form_fit`  -> ridge-regression solution of Eq. (8):
                          W* = (A^T A + lambda I)^-1 A^T y
  * standard `forward` + an external SGD training loop (Algorithm 1 of the
    paper), see trainer.py.

Categorical inputs (station, day-of-week, ...) are embedded before entering
the hidden layer. This is a practical extension the paper does not need
(it never scales to a shared model across thousands of trains); the
embeddings are frozen too when the closed-form solver is used, so that
solver still matches classic SELM exactly (only the output layer is fit).
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
}


class SELM(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        cardinalities: dict[str, int],
        embedding_dim: int = 8,
        hidden_dim: int = 512,
        activation: str = "tanh",
        hidden_init: str = "uniform",
        hidden_init_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATIONS)}")
        self.activation = _ACTIVATIONS[activation]
        self.categorical_cols = list(cardinalities.keys())

        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(card, embedding_dim) for col, card in cardinalities.items()}
        )
        input_dim = numeric_dim + embedding_dim * len(cardinalities)

        # Eq. (2): the random hidden layer. Weights are drawn once and frozen.
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=True)
        self._init_hidden_layer(hidden_init, hidden_init_range)
        for p in self.hidden.parameters():
            p.requires_grad = False

        # Eq. (3)/Fig. 4: output layer, no bias, this is the only thing SELM trains.
        self.output = nn.Linear(hidden_dim, 1, bias=False)

    def _init_hidden_layer(self, hidden_init: str, hidden_init_range: tuple[float, float]) -> None:
        if hidden_init == "uniform":
            lo, hi = hidden_init_range
            nn.init.uniform_(self.hidden.weight, lo, hi)
            nn.init.uniform_(self.hidden.bias, lo, hi)
        elif hidden_init == "normal":
            std = hidden_init_range[1]
            nn.init.normal_(self.hidden.weight, mean=0.0, std=std)
            nn.init.normal_(self.hidden.bias, mean=0.0, std=std)
        else:
            raise ValueError(f"Unknown hidden_init '{hidden_init}', choose 'uniform' or 'normal'")

    def freeze_embeddings(self) -> None:
        """Used by the closed_form solver: keep embeddings random/frozen so
        the *only* fitted parameters are the output weights, as in Eq. (8).
        """
        for p in self.embeddings.parameters():
            p.requires_grad = False

    def _build_input(self, x_num: torch.Tensor, x_cat: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [x_num]
        for col in self.categorical_cols:
            parts.append(self.embeddings[col](x_cat[col]))
        return torch.cat(parts, dim=1)

    def activations(self, x_num: torch.Tensor, x_cat: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns A (Eq. 4): the hidden-layer activation matrix for a batch."""
        x = self._build_input(x_num, x_cat)
        return self.activation(self.hidden(x))

    def forward(self, x_num: torch.Tensor, x_cat: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self.activations(x_num, x_cat)
        return self.output(a)

    @torch.no_grad()
    def closed_form_fit(self, loader, ridge_lambda: float, device: torch.device) -> None:
        """Solves Eq. (8): W* = (A^T A + lambda I)^-1 A^T y over the full
        training set, accumulated batch-by-batch to bound memory use.
        """
        h = self.hidden.out_features
        ata = torch.zeros(h, h, device=device)
        aty = torch.zeros(h, 1, device=device)
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device)
            x_cat = {k: v.to(device) for k, v in x_cat.items()}
            y = y.to(device)
            a = self.activations(x_num, x_cat)
            ata += a.T @ a
            aty += a.T @ y
        reg = ridge_lambda * torch.eye(h, device=device)
        w_star = torch.linalg.solve(ata + reg, aty)
        self.output.weight.copy_(w_star.T)
