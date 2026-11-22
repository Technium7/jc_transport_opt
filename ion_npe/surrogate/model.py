"""Encoder plus mixture head, and the numpy prediction API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from torch import nn

from .encoders import ResidualMLP, StructuredTransformer
from .factorized import (
    ArithmeticMeanHead,
    GammaShapeHead,
    MeanUncertaintyHead,
    ShapeHead,
    gamma_ratio_variance,
    predictive_nll,
    ratio_variance,
    sample_mixture_ratios,
    sample_unit_gamma,
)
from .heads import (
    MixtureHead,
    combine_members,
    diagonal_variance,
    lognormal_mixture_moments,
    mixture_log_density,
    mixture_log_moments,
    mixture_nll,
    sample_mixture,
)
from .torch_compat import tensor_to_numpy, to_tensor
from .layout import (
    E_REF_DEFAULT,
    JOINT_DIM,
    LOG_EPS,
    field_features,
    log_to_quanta,
    normalize_design,
    quanta_to_log,
)


@dataclass
class SurrogateConfig:
    arch: str = "transformer"
    n_components: int = 3
    dropout: float = 0.1
    hidden: int = 128
    n_blocks: int = 3
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff: int = 128
    var_floor: float = 1e-3
    e_ref: float = E_REF_DEFAULT
    log_eps: float = LOG_EPS
    head: str = "log_mixture"
    # Factorized heads only. The log-mixture head ignores these weights.
    lambda_mean: float = 1.0
    lambda_typical: float = 1.0
    lambda_scale: float = 1.0
    lambda_rank: float = 0.5
    lambda_shape: float = 0.2
    lambda_cloud: float = 0.5
    scale_floor: float = 1e-3
    copula: bool = False

    def __post_init__(self):
        if self.arch not in ("mlp", "transformer"):
            raise ValueError("arch must be mlp or transformer")
        if int(self.n_components) < 1:
            raise ValueError("n_components must be positive")
        if self.head not in ("log_mixture", "gauss_lognorm", "gamma"):
            raise ValueError("head must be log_mixture, gauss_lognorm, or gamma")
        if self.copula and self.head != "gauss_lognorm":
            raise ValueError("the Gaussian copula is only defined for the gauss_lognorm head")
        for name in ("lambda_mean", "lambda_typical", "lambda_scale", "lambda_rank", "lambda_shape", "lambda_cloud"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("{} must be non-negative".format(name))


class SurrogateNet(nn.Module):
    """One deterministic encoder and one conditional mixture head."""

    def __init__(self, config: SurrogateConfig):
        super().__init__()
        self.config = config
        if config.arch == "mlp":
            self.encoder = ResidualMLP(
                JOINT_DIM + 2,
                hidden=config.hidden,
                n_blocks=config.n_blocks,
                dropout=config.dropout,
            )
        else:
            self.encoder = StructuredTransformer(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                ff=config.ff,
                dropout=config.dropout,
            )
        self.head = None
        self.mean_head = None
        self.shape_head = None
        self.scale_head = None
        if config.head == "log_mixture":
            self.head = MixtureHead(
                self.encoder.out_dim,
                n_components=config.n_components,
                var_floor=config.var_floor,
            )
        else:
            self.mean_head = ArithmeticMeanHead(self.encoder.out_dim)
            self.scale_head = MeanUncertaintyHead(self.encoder.out_dim + 2)
            if config.head == "gamma":
                self.shape_head = GammaShapeHead(scale_floor=config.scale_floor)
            else:
                self.shape_head = ShapeHead(scale_floor=config.scale_floor, copula=config.copula)

    def forward(self, design, field):
        hidden = self.encoder(design, field)
        if self.config.head == "log_mixture":
            return self.head(hidden)
        mean, typical = self.mean_head(hidden, field)
        shared = {"mean": mean, "typical": typical, "mean_scale": self.scale_head(hidden, field)}
        if self.config.head == "gamma":
            shared["shape"] = self.shape_head(field)
            return shared
        gaussian_scale, lognormal_scale, logit, corr = self.shape_head(field)
        shared.update({
            "gaussian_scale": gaussian_scale,
            "lognormal_scale": lognormal_scale,
            "logit": logit,
            "corr": corr,
        })
        return shared


class SurrogatePredictor:
    """Normalized joint design and a stray-field amplitude in, heating distribution out."""

    def __init__(self, model, lower, upper, e_ref, log_eps, device="cpu"):
        self.model = model
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.e_ref = float(e_ref)
        self.log_eps = float(log_eps)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.members = None

    @property
    def config(self):
        return self.model.config

    def distribution(self, design, sigma):
        """Mixture parameters. ``log_pi`` is normalized."""
        design, sigma = _batch(design, sigma)
        self.model.eval()
        design_n = normalize_design(design, self.lower, self.upper)
        field = field_features(sigma, self.e_ref)
        with torch.no_grad():
            logits, mu, factor = self.model(
                to_tensor(design_n, dtype=torch.float32, device=self.device),
                to_tensor(field, dtype=torch.float32, device=self.device),
            )
            log_pi = torch.log_softmax(logits, dim=-1)
        return (
            tensor_to_numpy(log_pi.detach().cpu()),
            tensor_to_numpy(mu.detach().cpu()),
            tensor_to_numpy(factor.detach().cpu()),
        )

    def sample_log(self, design, sigma, n_samples=32, rng=None, include_mean_uncertainty=True):
        if self._factorized():
            quanta = self.sample_quanta(
                design, sigma, n_samples, rng, include_mean_uncertainty=include_mean_uncertainty
            )
            return np.log10(quanta + self.log_eps)
        log_pi, mu, factor = self.distribution(design, sigma)
        return sample_mixture(log_pi, mu, factor, int(n_samples), _rng(rng))

    def sample_quanta(self, design, sigma, n_samples=32, rng=None, include_mean_uncertainty=True):
        if self._factorized():
            return self._sample_factorized(
                design, sigma, int(n_samples), _rng(rng), include_mean_uncertainty=include_mean_uncertainty
            )
        return log_to_quanta(self.sample_log(design, sigma, n_samples, rng), self.log_eps)

    def log_moments(self, design, sigma):
        if self._factorized():
            logs = self.sample_log(design, sigma, n_samples=48, rng=12345)
            return logs.mean(axis=0), logs.var(axis=0)
        log_pi, mu, factor = self.distribution(design, sigma)
        pi = np.exp(log_pi)
        variance = diagonal_variance(factor)
        return mixture_log_moments(pi, mu, variance)

    def linear_moments(self, design, sigma):
        if self._factorized():
            params = self._factor_params(design, sigma)
            mean = params["mean"]
            if self.config.head == "gamma":
                var = mean ** 2 * gamma_ratio_variance(params["shape"])
            else:
                weight = 1.0 / (1.0 + np.exp(-params["logit"]))
                var = mean ** 2 * ratio_variance(params["gaussian_scale"], params["lognormal_scale"], weight)
            return mean, var
        log_pi, mu, factor = self.distribution(design, sigma)
        pi = np.exp(log_pi)
        variance = diagonal_variance(factor)
        return lognormal_mixture_moments(pi, mu, variance, self.log_eps)

    def arithmetic_mean(self, design, sigma):
        """Predicted arithmetic mean quanta. Log-mixture models use their linear moment."""
        mean, _var = self.linear_moments(design, sigma)
        return mean

    def expected_total(self, design, sigma):
        """Expected sum of the three motional quanta, in linear space."""
        mean, _var = self.linear_moments(design, sigma)
        return np.sum(mean, axis=-1)

    def total_samples(self, design, sigma, n_samples=256, rng=None, include_mean_uncertainty=False):
        """Sum of mode quanta. Default draws are the seed cloud around the predicted mean."""
        quanta = self.sample_quanta(
            design, sigma, n_samples, rng, include_mean_uncertainty=include_mean_uncertainty
        )
        return np.sum(quanta, axis=-1)

    def total_quantiles(self, design, sigma, probs, n_samples=256, rng=None):
        samples = self.total_samples(design, sigma, n_samples, rng)
        return np.quantile(samples, np.asarray(probs, dtype=np.float64), axis=0).T

    def cvar(self, design, sigma, alpha=0.9, n_samples=256, rng=None):
        """Mean total heating on the upper ``1 - alpha`` tail."""
        samples = self.total_samples(design, sigma, n_samples, rng)
        level = np.quantile(samples, float(alpha), axis=0)
        tail = samples >= level[None, :]
        counts = np.maximum(tail.sum(axis=0), 1)
        return (samples * tail).sum(axis=0) / counts

    def negative_log_likelihood(self, design, sigma, quanta, valid):
        """Per-pair mean NLL. Invalid realizations are masked, not replaced.

        Factorized heads return the linear-phonon NLL. Zero field is excluded
        from that density and comes back as NaN.
        """
        design, sigma = _batch(design, sigma)
        self.model.eval()
        design_n = normalize_design(design, self.lower, self.upper)
        field = field_features(sigma, self.e_ref)
        mask = np.asarray(valid, dtype=np.bool_)
        with torch.no_grad():
            pred = self.model(
                to_tensor(design_n, dtype=torch.float32, device=self.device),
                to_tensor(field, dtype=torch.float32, device=self.device),
            )
            if self._factorized():
                raw = np.nan_to_num(np.asarray(quanta, dtype=np.float64), nan=0.0)
                nll = predictive_nll(
                    pred,
                    to_tensor(raw, dtype=torch.float64, device=self.device),
                    to_tensor(mask, dtype=torch.bool, device=self.device),
                )
                scores = tensor_to_numpy(nll.detach().cpu())
                scores = np.where(np.asarray(sigma) > 0.0, scores, np.nan)
                return scores
            target = np.nan_to_num(quanta_to_log(quanta, self.log_eps), nan=0.0)
            logits, mu, factor = pred
            _loss, per_pair = mixture_nll(
                logits,
                mu,
                factor,
                to_tensor(target, dtype=torch.float32, device=self.device),
                to_tensor(mask, dtype=torch.bool, device=self.device),
            )
        return tensor_to_numpy(per_pair.detach().cpu())

    def linear_negative_log_likelihood(self, design, sigma, quanta, valid):
        """Per-pair NLL of the linear phonon vector, with the log10 Jacobian.

        The log-mixture head is converted from its log10 density. Exact zeros
        are left out of the average instead of being moved to ``log10(eps)``.
        Factorized heads already score linear quanta; zero field stays NaN.
        """
        if self._factorized():
            return self.negative_log_likelihood(design, sigma, quanta, valid)
        design, sigma = _batch(design, sigma)
        quanta = np.asarray(quanta, dtype=np.float64)
        mask = np.asarray(valid, dtype=bool) & np.isfinite(quanta).all(axis=-1) & (quanta > 0.0).all(axis=-1)
        self.model.eval()
        design_n = normalize_design(design, self.lower, self.upper)
        field = field_features(sigma, self.e_ref)
        y = np.zeros(quanta.shape, dtype=np.float64)
        y[mask] = np.log10(quanta[mask] + self.log_eps)
        with torch.no_grad():
            logits, mu, factor = self.model(
                to_tensor(design_n, dtype=torch.float32, device=self.device),
                to_tensor(field, dtype=torch.float32, device=self.device),
            )
            log_y = mixture_log_density(
                logits,
                mu,
                factor,
                to_tensor(y, dtype=torch.float32, device=self.device),
            )
        log_y = tensor_to_numpy(log_y.detach().cpu())
        jac = np.sum(np.log((np.maximum(quanta, 0.0) + self.log_eps) * np.log(10.0)), axis=-1)
        log_n = log_y - jac
        scores = np.full(len(design), np.nan)
        for index in range(len(design)):
            chosen = mask[index]
            if np.any(chosen):
                scores[index] = float(-np.mean(log_n[index, chosen]))
        return scores

    def _factorized(self):
        return self.config.head != "log_mixture"

    def _factor_params(self, design, sigma):
        design, sigma = _batch(design, sigma)
        self.model.eval()
        design_n = normalize_design(design, self.lower, self.upper)
        field = field_features(sigma, self.e_ref)
        with torch.no_grad():
            pred = self.model(
                to_tensor(design_n, dtype=torch.float32, device=self.device),
                to_tensor(field, dtype=torch.float32, device=self.device),
            )
        return {key: None if value is None else tensor_to_numpy(value.detach().cpu()) for key, value in pred.items()}

    def shape_parameters(self, design, sigma):
        """Learned ratio shape. ``None`` for the log-quanta mixture head."""
        if not self._factorized():
            return None
        params = self._factor_params(design, sigma)
        report = {"mean_scale": params["mean_scale"], "typical": params["typical"]}
        if self.config.head == "gamma":
            report["shape"] = params["shape"]
            return report
        weight = 1.0 / (1.0 + np.exp(-params["logit"]))
        report.update({
            "gaussian_scale": params["gaussian_scale"],
            "lognormal_scale": params["lognormal_scale"],
            "lognormal_weight": weight,
        })
        if params.get("corr") is not None:
            raw = np.asarray(params["corr"], dtype=np.float64)
            r01 = np.tanh(raw[:, 0])
            r02 = np.tanh(raw[:, 1])
            bound = np.sqrt(np.maximum((1.0 - r01 ** 2) * (1.0 - r02 ** 2), 1e-12))
            r12 = r01 * r02 + bound * np.tanh(raw[:, 2])
            report["correlation"] = np.stack((r01, r02, r12), axis=-1)
        return report

    def ratio_cv(self, design, sigma):
        """Coefficient of variation of the within-design ratio, without mean uncertainty."""
        if not self._factorized():
            return None
        params = self._factor_params(design, sigma)
        if self.config.head == "gamma":
            variance = gamma_ratio_variance(params["shape"])
        else:
            weight = 1.0 / (1.0 + np.exp(-params["logit"]))
            variance = ratio_variance(params["gaussian_scale"], params["lognormal_scale"], weight)
        return np.sqrt(np.maximum(variance, 0.0))

    def _sample_factorized(self, design, sigma, n_samples, rng, include_mean_uncertainty=True):
        design, sigma = _batch(design, sigma)
        params = self._factor_params(design, sigma)
        mean = params["mean"]
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if self.config.head == "gamma":
            ratios = sample_unit_gamma(params["shape"], rng, (n_samples,) + mean.shape)
        else:
            weight = 1.0 / (1.0 + np.exp(-params["logit"]))
            ratios = sample_mixture_ratios(
                params["gaussian_scale"],
                params["lognormal_scale"],
                weight,
                rng,
                n_samples,
                corr=params.get("corr"),
            )
        samples = mean[None, :, :] * ratios
        if include_mean_uncertainty:
            shock = rng.normal(size=(samples.shape[0], mean.shape[0], 1))
            samples = samples * np.exp(np.asarray(params["mean_scale"], dtype=np.float64)[None, :, :] * shock)
        frozen = sigma <= 0.0
        if np.any(frozen):
            samples[:, frozen, :] = mean[None, frozen, :]
        return samples

    def predict_heating(self, design, sigma, n_samples=32, rng=None):
        """Distribution summary. ``sigma`` is the stray-field amplitude in V/m."""
        log_mean, log_var = self.log_moments(design, sigma)
        lin_mean, lin_var = self.linear_moments(design, sigma)
        total_mean = np.sum(lin_mean, axis=-1)
        samples = self.total_samples(design, sigma, n_samples, rng)
        return {
            "mean": lin_mean,
            "std": np.sqrt(lin_var),
            "total_mean": total_mean,
            "total_std": samples.std(axis=0),
            "log_mean": log_mean,
            "log_std": np.sqrt(log_var),
            "aleatoric_total_std": samples.std(axis=0),
            "epistemic_total_std": None,
        }

    def epistemic_total_std(self, design, sigma):
        return None

    def save(self, path, extra=None):
        payload = {
            "kind": "surrogate",
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "lower": self.lower,
            "upper": self.upper,
            "e_ref": self.e_ref,
            "log_eps": self.log_eps,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        allowed = {item.name for item in fields(SurrogateConfig)}
        config = SurrogateConfig(**{key: value for key, value in payload["config"].items() if key in allowed})
        model = SurrogateNet(config)
        model.load_state_dict(payload["state_dict"])
        if device is None:
            device = default_device()
        return cls(model, payload["lower"], payload["upper"], payload["e_ref"], payload["log_eps"], device)


class EnsemblePredictor(SurrogatePredictor):
    """Equal mixture of independently trained predictors. Disagreement is epistemic."""

    def __init__(self, members, device="cpu"):
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        first = members[0]
        super().__init__(first.model, first.lower, first.upper, first.e_ref, first.log_eps, device)
        self.members = [item for item in members]
        for member in self.members:
            if not np.allclose(member.lower, first.lower) or not np.allclose(member.upper, first.upper):
                raise ValueError("ensemble members disagree on normalization")
            if abs(member.e_ref - first.e_ref) > 0.0 or abs(member.log_eps - first.log_eps) > 0.0:
                raise ValueError("ensemble members disagree on field or quanta scaling")

    def distribution(self, design, sigma):
        parts = []
        for member in self.members:
            member.device = self.device
            member.model.to(self.device)
            parts.append(member.distribution(design, sigma))
        return combine_members(parts)

    def negative_log_likelihood(self, design, sigma, quanta, valid):
        log_pi, mu, factor = self.distribution(design, sigma)
        target = np.nan_to_num(quanta_to_log(quanta, self.log_eps), nan=0.0)
        mask = np.asarray(valid, dtype=np.bool_)
        # ``log_softmax(log pi)`` recovers ``log pi`` when ``pi`` sums to one.
        with torch.no_grad():
            _loss, per_pair = mixture_nll(
                to_tensor(log_pi, dtype=torch.float32),
                to_tensor(mu, dtype=torch.float32),
                to_tensor(factor, dtype=torch.float32),
                to_tensor(target, dtype=torch.float32),
                to_tensor(mask, dtype=torch.bool),
            )
        return tensor_to_numpy(per_pair.detach().cpu())

    def epistemic_total_std(self, design, sigma):
        totals = np.stack([member.expected_total(design, sigma) for member in self.members], axis=0)
        return totals.std(axis=0)

    def predict_heating(self, design, sigma, n_samples=32, rng=None):
        summary = super().predict_heating(design, sigma, n_samples=n_samples, rng=rng)
        summary["epistemic_total_std"] = self.epistemic_total_std(design, sigma)
        summary["aleatoric_total_std"] = summary["total_std"]
        return summary

    def save(self, directory, extra=None):
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(self.members):
            member.save(directory / "member_{}.pt".format(index), extra=extra if index == 0 else None)
        torch.save({"kind": "ensemble", "n_members": len(self.members)}, directory / "ensemble.pt")

    @classmethod
    def load(cls, directory, device=None):
        from pathlib import Path

        directory = Path(directory)
        payload = torch.load(directory / "ensemble.pt", map_location="cpu", weights_only=False)
        members = [
            SurrogatePredictor.load(directory / "member_{}.pt".format(index), device=device)
            for index in range(int(payload["n_members"]))
        ]
        if device is None:
            device = default_device()
        return cls(members, device)


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch(design, sigma):
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(1, -1)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(len(design), float(sigma))
    sigma = sigma.reshape(-1)
    if len(sigma) == 1 and len(design) > 1:
        sigma = np.full(len(design), float(sigma[0]))
    if len(sigma) != len(design):
        raise ValueError("designs and field amplitudes have different lengths")
    if design.shape[1] != JOINT_DIM:
        raise ValueError("expected {} joint parameters, got {}".format(JOINT_DIM, design.shape[1]))
    return design, sigma


def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng))
