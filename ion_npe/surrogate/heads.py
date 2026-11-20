"""Conditional Gaussian and Gaussian-mixture heads.

Covariance uses a Cholesky factor. Diagonal entries are softplus plus a floor,
so every component covariance is positive definite. The likelihood is the
log-sum-exp of the component densities.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_MODES = 3
_CHOL_PAIRS = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
_LN10 = math.log(10.0)


def _inv_softplus(value):
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


class MixtureHead(nn.Module):
    """Map a shared representation to a K-component Gaussian mixture on log quanta."""

    def __init__(self, in_dim, n_components=3, n_modes=N_MODES, var_floor=1e-3):
        super().__init__()
        self.n_components = int(n_components)
        self.n_modes = int(n_modes)
        self.var_floor = float(var_floor)
        if self.n_components < 1:
            raise ValueError("need at least one mixture component")
        if self.n_modes != 3:
            raise ValueError("this head is built for three motional modes")
        if self.var_floor <= 0.0:
            raise ValueError("variance floor must be positive")
        self.width = 1 + self.n_modes + len(_CHOL_PAIRS)
        self.proj = nn.Linear(int(in_dim), self.n_components * self.width)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data.mul_(0.1)
        nn.init.zeros_(self.proj.bias)
        target = max(0.5 - self.var_floor, 0.05)
        diag_bias = _inv_softplus(target)
        bias = self.proj.bias.data
        for component in range(self.n_components):
            base = component * self.width
            for slot, (row, col) in enumerate(_CHOL_PAIRS):
                if row == col:
                    bias[base + 1 + self.n_modes + slot] = diag_bias

    def set_mean_bias(self, mean_log):
        """Start every component mean at the training log-quanta center."""
        from .torch_compat import to_tensor

        mean_log = to_tensor(mean_log, dtype=self.proj.bias.dtype, device=self.proj.bias.device)
        if mean_log.shape != (self.n_modes,):
            raise ValueError("mean_log must have one value per mode")
        with torch.no_grad():
            for component in range(self.n_components):
                base = component * self.width + 1
                self.proj.bias[base:base + self.n_modes] = mean_log

    def forward(self, hidden):
        raw = self.proj(hidden).view(-1, self.n_components, self.width)
        logits = raw[..., 0]
        mu = raw[..., 1:1 + self.n_modes]
        factor = self._cholesky(raw[..., 1 + self.n_modes:])
        return logits, mu, factor

    def _cholesky(self, packed):
        eye = packed.new_zeros(packed.shape[:-1] + (self.n_modes, self.n_modes))
        for slot, (row, col) in enumerate(_CHOL_PAIRS):
            if row == col:
                eye[..., row, col] = F.softplus(packed[..., slot]) + self.var_floor
            else:
                eye[..., row, col] = packed[..., slot]
        return eye


def gaussian_log_prob(y, mu, factor):
    """Log density of ``N(mu, factor factor^T)``. ``factor`` is lower triangular."""
    batch = torch.broadcast_shapes(y.shape[:-1], mu.shape[:-1], factor.shape[:-2])
    y = y.expand(batch + (y.shape[-1],))
    mu = mu.expand(batch + (mu.shape[-1],))
    factor = factor.expand(batch + factor.shape[-2:])
    diff = (y - mu).reshape(-1, y.shape[-1], 1)
    solved = torch.linalg.solve_triangular(
        factor.reshape(-1, factor.shape[-1], factor.shape[-1]),
        diff,
        upper=False,
    ).squeeze(-1)
    quad = solved.square().sum(dim=-1).reshape(batch)
    log_det = factor.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(dim=-1)
    modes = y.shape[-1]
    return -0.5 * modes * math.log(2.0 * math.pi) - log_det - 0.5 * quad


def mixture_nll(logits, mu, factor, y, mask):
    """Mean over design-field pairs of the mean sample NLL. ``mask`` is ``(B, S)``."""
    log_pi = F.log_softmax(logits, dim=-1)
    log_comp = gaussian_log_prob(y[:, :, None, :], mu[:, None, :, :], factor[:, None, :, :, :])
    log_mix = torch.logsumexp(log_pi[:, None, :] + log_comp, dim=-1)
    mask = mask.to(dtype=log_mix.dtype)
    counted = mask.sum(dim=-1).clamp_min(1.0)
    per_pair = -(log_mix * mask).sum(dim=-1) / counted
    keep = mask.sum(dim=-1) > 0
    if not bool(keep.any()):
        raise ValueError("mixture NLL received no valid samples")
    return per_pair[keep].mean(), per_pair


def component_weights(logits):
    return torch.softmax(logits, dim=-1)


def sample_mixture(log_pi, mu, factor, n_samples, rng):
    """Draw ``(n_samples, batch, modes)`` log-quanta. ``log_pi`` is already normalized."""
    log_pi = np.asarray(log_pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    pi = np.exp(log_pi - log_pi.max(axis=-1, keepdims=True))
    pi = pi / pi.sum(axis=-1, keepdims=True)
    batch, components, modes = mu.shape
    cdf = np.cumsum(pi, axis=-1)
    draw = rng.random((int(n_samples), batch))
    choice = (cdf[None, :, :] < draw[..., None]).sum(axis=-1)
    choice = np.clip(choice, 0, components - 1)
    eps = rng.normal(size=(int(n_samples), batch, modes))
    batch_index = np.arange(batch)[None, :]
    chosen_mu = mu[batch_index, choice]
    chosen = factor[batch_index, choice]
    return chosen_mu + np.einsum("nbij,nbj->nbi", chosen, eps)


def mixture_log_moments(pi, mu, variance):
    """Mean and variance of each mode under the mixture, in log-quanta space."""
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 0.0)
    mean = np.sum(pi[..., None] * mu, axis=-2)
    second = np.sum(pi[..., None] * (variance + mu ** 2), axis=-2)
    var = np.maximum(second - mean ** 2, 0.0)
    return mean, var


def lognormal_mixture_moments(pi, mu, variance, eps):
    """Mean and variance of ``10**y - eps`` for a mixture of diagonal log10-Gaussians.

    Expectation is taken in linear quanta. It is not ``10`` to the power of
    the mean log-quanta. Correlation across modes does not change this
    per-mode expectation; quantiles of the sum still need samples.
    """
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 0.0)
    mean_shift = np.exp(mu * _LN10 + 0.5 * variance * _LN10 ** 2)
    var_shift = np.maximum(mean_shift ** 2 * (np.exp(variance * _LN10 ** 2) - 1.0), 0.0)
    mean_k = mean_shift - float(eps)
    # n = s - eps, so Var(n) = Var(s) and E[n^2] = Var(s) + (E[s] - eps)^2
    # only when we do not clip. The closed form is the unclipped inverse.
    second_k = var_shift + mean_k ** 2
    mean = np.sum(pi[..., None] * mean_k, axis=-2)
    second = np.sum(pi[..., None] * second_k, axis=-2)
    var = np.maximum(second - mean ** 2, 0.0)
    return mean, var


def combine_members(members):
    """Uniform mixture of ensemble members. Each item is ``(log_pi, mu, factor)``."""
    logs = []
    mus = []
    factors = []
    scale = math.log(len(members))
    for log_pi, mu, factor in members:
        logs.append(np.asarray(log_pi, dtype=np.float64) - scale)
        mus.append(np.asarray(mu, dtype=np.float64))
        factors.append(np.asarray(factor, dtype=np.float64))
    return np.concatenate(logs, axis=-1), np.concatenate(mus, axis=-2), np.concatenate(factors, axis=-3)


def diagonal_variance(factor):
    """Diagonal of ``L L^T`` from a lower-triangular factor."""
    factor = np.asarray(factor, dtype=np.float64)
    return np.sum(factor ** 2, axis=-1)
