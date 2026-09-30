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


@torch.no_grad()
def _ridge_closed_form(a: torch.Tensor, y: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    """Eq. (8), generalized to a matrix target Y (n x d_out) instead of a
    vector y (n x 1) -- the normal equation is the same either way:
    beta = (A^T A + lambda I)^-1 A^T Y, beta: (h x d_out).
    """
    h = a.shape[1]
    reg = ridge_lambda * torch.eye(h, device=a.device)
    return torch.linalg.solve(a.T @ a + reg, a.T @ y)


class DELM(nn.Module):
    """Deep Extreme Learning Machine (Oneto et al. 2018, Section 3.2).

    Implements the simpler architecture the paper actually uses, from Kasun
    et al. [36] ("Representational learning with ELMs for big data") --
    NOT the backprop-finetuned variant of Tang et al. [39], which the paper
    explicitly avoids ("DELM do not require fine-tuning for the entire
    system", since it "requires more complex and time consuming
    computations").

    Each of the l layers is an ELM autoencoder: a random hidden projection
    (same Eq. 2 as SELM) whose output weights are solved in closed form
    (Eq. 8, generalized to a matrix target -- see `_ridge_closed_form`) to
    RECONSTRUCT that layer's own input, instead of predicting y. The
    *learned* reconstruction weights beta_i (not the random ones) become
    the deployed encoder for that layer: X_i = phi(X_{i-1} @ beta_i^T)
    (Eq. 9). Stacking l such layers distills the raw input into a deep
    representation X_l, which is then fed -- "without random feature
    mapping" -- into a single trainable linear output layer, exactly like
    SELM's own output layer (same Fig. 4 idea, just on top of X_l instead
    of a single random hidden layer).

    `fit_representation()` performs this l-layer closed-form pretraining
    once, over the full training set; there is no gradient-based
    alternative for it in the paper. Only the final output layer (and the
    embeddings) are then trained by Trainer.fit_sgd, exactly as for SELM --
    this project doesn't use the closed_form solver for the final layer,
    so it isn't wired up here (unlike SELM, which supports both).

    Simplifications vs. the paper (documented, not hidden): the paper found
    (Table 4) hyperparameters like l~6 and h_i in the hundreds, but that was
    tuned on RFI's much higher-dimensional multi-train feature space; ours
    (~24-39 input dims) is far smaller, so `hidden_dims` defaults to a small
    funnel instead. Only one shared `ridge_lambda` is used across all AE
    layers, rather than one per layer.
    """

    def __init__(
        self,
        numeric_dim: int,
        cardinalities: dict[str, int],
        embedding_dim: int = 8,
        hidden_dims: list[int] = (64, 32, 16),
        activation: str = "tanh",
        hidden_init: str = "uniform",
        hidden_init_range: tuple[float, float] = (-1.0, 1.0),
        ridge_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATIONS)}")
        if not hidden_dims:
            raise ValueError("DELM needs at least one layer in hidden_dims")
        self.activation = _ACTIVATIONS[activation]
        self.categorical_cols = list(cardinalities.keys())
        self.hidden_dims = list(hidden_dims)
        self.ridge_lambda = ridge_lambda
        self.hidden_init = hidden_init
        self.hidden_init_range = hidden_init_range

        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(card, embedding_dim) for col, card in cardinalities.items()}
        )
        layer_input_dim = numeric_dim + embedding_dim * len(cardinalities)

        # beta_i (h_i x d_{i-1}): each AE layer's learned reconstruction
        # weights, solved once by fit_representation() -- frozen (like
        # SELM.hidden), nothing ever backprops through them.
        self.encoders = nn.ParameterList()
        for h in self.hidden_dims:
            self.encoders.append(nn.Parameter(torch.zeros(h, layer_input_dim), requires_grad=False))
            layer_input_dim = h

        self.output = nn.Linear(self.hidden_dims[-1], 1, bias=False)
        self._fitted = False

    def load_state_dict(self, state_dict, strict: bool = True):
        """A loaded checkpoint's `encoders` were necessarily produced by a
        completed fit_representation() call, but `_fitted` is a plain
        Python flag, not a tensor -- it doesn't travel through state_dict.
        Without this override, reloading a saved DELM in a fresh process
        (e.g. for analysis or inference) would hit the activations() guard
        below despite the encoders already being valid.
        """
        result = super().load_state_dict(state_dict, strict=strict)
        self._fitted = True
        return result

    def freeze_embeddings(self) -> None:
        for p in self.embeddings.parameters():
            p.requires_grad = False

    def _build_input(self, x_num: torch.Tensor, x_cat: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [x_num]
        for col in self.categorical_cols:
            parts.append(self.embeddings[col](x_cat[col]))
        return torch.cat(parts, dim=1)

    def _random_hidden_layer(self, in_dim: int, out_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hidden_init == "uniform":
            lo, hi = self.hidden_init_range
            w = torch.empty(out_dim, in_dim, device=device).uniform_(lo, hi)
            b = torch.empty(out_dim, device=device).uniform_(lo, hi)
        elif self.hidden_init == "normal":
            std = self.hidden_init_range[1]
            w = torch.empty(out_dim, in_dim, device=device).normal_(0.0, std)
            b = torch.empty(out_dim, device=device).normal_(0.0, std)
        else:
            raise ValueError(f"Unknown hidden_init '{self.hidden_init}', choose 'uniform' or 'normal'")
        return w, b

    @torch.no_grad()
    def fit_representation(self, loader, device: torch.device) -> None:
        """Greedily pretrains the l ELM-autoencoder layers in closed form,
        one layer at a time, over the FULL training set collected from
        `loader`. Must be called once before fit_sgd -- activations()
        raises until this has run.
        """
        self.to(device)
        x_num_batches, x_cat_batches = [], []
        for batch in loader:
            x_num_batches.append(batch[0])
            x_cat_batches.append(batch[1])
        x_num_full = torch.cat(x_num_batches).to(device)
        x_cat_full = {
            col: torch.cat([c[col] for c in x_cat_batches]).to(device) for col in self.categorical_cols
        }

        x = self._build_input(x_num_full, x_cat_full)
        for i, h in enumerate(self.hidden_dims):
            in_dim = x.shape[1]
            w, b = self._random_hidden_layer(in_dim, h, device)
            hidden = self.activation(x @ w.T + b)
            beta = _ridge_closed_form(hidden, x, self.ridge_lambda)  # (h, in_dim), reconstructs x
            self.encoders[i].data = beta
            x = self.activation(x @ beta.T)  # deployed encoder for this layer -- Eq. (9)
        self._fitted = True

    def activations(self, x_num: torch.Tensor, x_cat: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns X_l: the deep representation after the l AE layers."""
        if not self._fitted:
            raise RuntimeError("DELM.fit_representation() must be called before activations()/forward().")
        x = self._build_input(x_num, x_cat)
        for beta in self.encoders:
            x = self.activation(x @ beta.T)
        return x

    def forward(self, x_num: torch.Tensor, x_cat: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.output(self.activations(x_num, x_cat))
