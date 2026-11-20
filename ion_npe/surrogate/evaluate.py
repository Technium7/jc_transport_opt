"""Shared metrics for the original stochastic MLP and the new surrogates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .layout import quanta_to_log


def evaluate_predictor(
    predictor,
    table,
    indices,
    n_mc=128,
    low_total=1e4,
    extreme_total=1e8,
    rng=0,
    ranking_k=10,
):
    """Score one split. Large finite heating is kept. Invalid realizations are skipped."""
    indices = np.asarray(indices, dtype=int)
    design = table.design[indices]
    sigma = table.sigma[indices]
    quanta = table.quanta[indices]
    valid = table.valid[indices]
    true_log = quanta_to_log(quanta, predictor.log_eps)
    true_log_mean = _masked_mean(true_log, valid)
    pred_log_mean, _log_var = predictor.log_moments(design, sigma)
    per_nll = predictor.negative_log_likelihood(design, sigma, quanta, valid)
    keep = valid.any(axis=1)
    per_nll = np.where(keep, per_nll, np.nan)
    samples = predictor.sample_log(design, sigma, n_samples=int(n_mc), rng=rng)
    expected = predictor.expected_total(design, sigma)
    true_total = _true_total(quanta, valid)
    metrics = {
        "n": int(keep.sum()),
        "nll": _nanmean(per_nll),
        "log_rmse": _rmse(pred_log_mean[keep], true_log_mean[keep]),
        "log_mae": _mae(pred_log_mean[keep], true_log_mean[keep]),
        "coverage_50": _coverage(samples, true_log, valid, 0.25, 0.75),
        "coverage_90": _coverage(samples, true_log, valid, 0.05, 0.95),
        "total_mae": _mae(expected[keep], true_total[keep]),
        "total_median_ae": _median_ae(expected[keep], true_total[keep]),
        "log_total_mae": _mae(np.log10(np.maximum(expected[keep], 1e-30)), np.log10(np.maximum(true_total[keep], 1e-30))),
        "quantile_log_mae": _quantile_log_mae(samples, true_log, valid, quanta, (0.1, 0.5, 0.9)),
    }
    metrics["by_field"] = _by_field(
        sigma, keep, pred_log_mean, true_log_mean, per_nll, expected, true_total, samples, true_log, valid
    )
    metrics["regimes"] = {
        "low": _subset_metrics(true_total <= float(low_total), keep, pred_log_mean, true_log_mean, expected, true_total),
        "extreme": _subset_metrics(true_total >= float(extreme_total), keep, pred_log_mean, true_log_mean, expected, true_total),
    }
    metrics["ranking"] = rank_metrics(sigma[keep], expected[keep], true_total[keep], k=ranking_k)
    metrics["curves"] = {
        "sigma": sigma,
        "true_log_mean": true_log_mean,
        "pred_log_mean": pred_log_mean,
        "true_total": true_total,
        "expected_total": expected,
        "valid": keep,
    }
    return metrics


def rank_metrics(sigma, predicted, truth, k=10):
    """How well predicted heating orders designs, per stray-field amplitude."""
    rows = []
    for label in np.unique(np.round(sigma, 8)):
        chosen = np.round(sigma, 8) == label
        pred = np.asarray(predicted)[chosen]
        true = np.asarray(truth)[chosen]
        if pred.size == 0 or not np.isfinite(true).any():
            continue
        order_pred = np.argsort(pred)
        order_true = np.argsort(true)
        take = min(int(k), pred.size)
        overlap = len(set(order_pred[:take].tolist()) & set(order_true[:take].tolist())) / float(take)
        best = int(order_pred[0])
        rows.append({
            "sigma": float(label),
            "n": int(pred.size),
            "predicted_best_true_total": float(true[best]),
            "true_best_total": float(np.min(true)),
            "log10_regret": float(np.log10(max(true[best], 1e-30)) - np.log10(max(np.min(true), 1e-30))),
            "top_overlap": float(overlap),
            "spearman_log": _spearman(np.log10(np.maximum(pred, 1e-30)), np.log10(np.maximum(true, 1e-30))),
        })
    return rows


def fit_legacy(split, epochs=20, hidden=48, seed=0):
    """Train the original stochastic MLP on design vectors only, where it applies.

    Rows need at least two finite realizations. The field amplitude is not an
    input, which is the comparison the new models are meant to improve on.
    """
    from ion_npe.train import train_predictor

    designs = []
    samples = []
    for index in split.train:
        mask = split.table.valid[index]
        if int(mask.sum()) < 2:
            continue
        designs.append(split.table.design[index])
        samples.append(split.table.quanta[index, mask])
    if len(samples) < 4:
        return None
    width = min(item.shape[0] for item in samples)
    if width < 2:
        return None
    quanta = np.stack([item[:width] for item in samples], axis=0)
    design = np.vstack(designs)
    model, _history = train_predictor(
        design,
        quanta,
        kind="stochastic",
        hidden=int(hidden),
        epochs=int(epochs),
        seed=int(seed),
        log_every=0,
        lbfgs_maxiter=0,
        noise_samples=8,
        val_fraction=0.15,
    )
    return model


def evaluate_legacy(model, table, indices, n_mc=64, low_total=1e4, extreme_total=1e8, rng=0, ranking_k=10):
    """Score the original network with the same summaries, using its Monte Carlo cloud."""
    indices = np.asarray(indices, dtype=int)
    design = table.design[indices]
    sigma = table.sigma[indices]
    quanta = table.quanta[indices]
    valid = table.valid[indices]
    logs = model.predict_log_samples(design, n_samples=int(n_mc), rng=np.random.default_rng(rng))
    pred_log_mean = logs.mean(axis=0)
    true_log = quanta_to_log(quanta)
    true_log_mean = _masked_mean(true_log, valid)
    keep = valid.any(axis=1)
    per_nll = _gaussian_nll(logs, true_log, valid)
    expected = np.mean(np.sum(np.power(10.0, logs), axis=-1), axis=0)
    true_total = _true_total(quanta, valid)
    metrics = {
        "n": int(keep.sum()),
        "nll": _nanmean(per_nll),
        "nll_kind": "gaussian_on_monte_carlo",
        "log_rmse": _rmse(pred_log_mean[keep], true_log_mean[keep]),
        "log_mae": _mae(pred_log_mean[keep], true_log_mean[keep]),
        "coverage_50": _coverage(logs, true_log, valid, 0.25, 0.75),
        "coverage_90": _coverage(logs, true_log, valid, 0.05, 0.95),
        "total_mae": _mae(expected[keep], true_total[keep]),
        "total_median_ae": _median_ae(expected[keep], true_total[keep]),
        "log_total_mae": _mae(np.log10(np.maximum(expected[keep], 1e-30)), np.log10(np.maximum(true_total[keep], 1e-30))),
        "quantile_log_mae": _quantile_log_mae(logs, true_log, valid, quanta, (0.1, 0.5, 0.9)),
        "by_field": _by_field(
            sigma, keep, pred_log_mean, true_log_mean, per_nll, expected, true_total, logs, true_log, valid
        ),
        "regimes": {
            "low": _subset_metrics(true_total <= float(low_total), keep, pred_log_mean, true_log_mean, expected, true_total),
            "extreme": _subset_metrics(true_total >= float(extreme_total), keep, pred_log_mean, true_log_mean, expected, true_total),
        },
        "ranking": rank_metrics(sigma[keep], expected[keep], true_total[keep], k=ranking_k),
        "curves": {
            "sigma": sigma,
            "true_log_mean": true_log_mean,
            "pred_log_mean": pred_log_mean,
            "true_total": true_total,
            "expected_total": expected,
            "valid": keep,
        },
    }
    return metrics


def public_metrics(metrics):
    """JSON-safe metrics without the plotting arrays."""
    skip = {"curves"}
    out = {}
    for key, value in metrics.items():
        if key in skip:
            continue
        out[key] = _jsonify(value)
    return out


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public_metrics(metrics), indent=2))
    return path


def plot_training(history, path, title="training"):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(history["train"], label="train", color="#1f4e79")
    if history.get("val"):
        ax.plot(history["val"], label="validation", color="#c47b2b")
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log-likelihood")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_prediction(metrics, path, title="predicted and simulated log quanta"):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    curves = metrics["curves"]
    keep = np.asarray(curves["valid"], dtype=bool)
    true = np.asarray(curves["true_log_mean"])[keep].reshape(-1)
    pred = np.asarray(curves["pred_log_mean"])[keep].reshape(-1)
    sigma = np.repeat(np.asarray(curves["sigma"])[keep], 3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    scatter = ax.scatter(true, pred, c=np.log10(1.0 + sigma), s=8, cmap="viridis", alpha=0.7, linewidths=0)
    limit = [np.nanmin([true.min(), pred.min()]), np.nanmax([true.max(), pred.max()])]
    ax.plot(limit, limit, color="#444444", lw=1.0)
    ax.set_xlabel("simulated mean log10 quanta")
    ax.set_ylabel("predicted mean log10 quanta")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="log10(1 + V/m)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_field_errors(rows, path):
    """``rows`` is a list of ``(name, by_field metrics)``."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name, by_field in rows:
        sigma = [row["sigma"] for row in by_field]
        error = [row["log_rmse"] for row in by_field]
        ax.plot(sigma, error, marker="o", label=name)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("stray field (V/m)")
    ax.set_ylabel("RMSE of mean log10 quanta")
    ax.set_title("error by stray-field amplitude")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_coverage(rows, path):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _metrics in rows]
    cover_50 = [metrics["coverage_50"] for _name, metrics in rows]
    cover_90 = [metrics["coverage_90"] for _name, metrics in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - 0.16, cover_50, width=0.32, label="50% interval", color="#1f4e79")
    ax.bar(x + 0.16, cover_90, width=0.32, label="90% interval", color="#7aa6c2")
    ax.axhline(0.50, color="#1f4e79", lw=0.8, ls="--")
    ax.axhline(0.90, color="#7aa6c2", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("coverage")
    ax.set_title("predictive interval coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_distributions(predictor, table, indices, path, n_mc=256, rng=0):
    """Histograms of total heating for a few held-out design-field pairs."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    indices = np.asarray(list(indices), dtype=int)
    if indices.size == 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = predictor.sample_quanta(
        table.design[indices], table.sigma[indices], n_samples=int(n_mc), rng=rng
    )
    pred_total = np.sum(samples, axis=-1)
    fig, axes = plt.subplots(1, indices.size, figsize=(3.3 * indices.size, 3.2), squeeze=False)
    for panel, index in enumerate(indices):
        ax = axes[0, panel]
        truth = np.sum(table.quanta[index, table.valid[index]], axis=-1)
        ax.hist(np.log10(np.maximum(truth, 1e-30)), bins=12, density=True, color="#c5c5c5", label="simulated")
        ax.hist(
            np.log10(np.maximum(pred_total[:, panel], 1e-30)),
            bins=12, density=True, histtype="step", color="#1f4e79", lw=1.6, label="predicted",
        )
        ax.set_title("{:.3g} V/m".format(float(table.sigma[index])))
        ax.set_xlabel("log10 total quanta")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ranking(metrics, path, title="candidate ranking"):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    rows = metrics["ranking"]
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sigma = [row["sigma"] for row in rows]
    regret = [row["log10_regret"] for row in rows]
    overlap = [row["top_overlap"] for row in rows]
    ax.plot(sigma, regret, marker="o", color="#8c2f39", label="log10 regret of predicted best")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("stray field (V/m)")
    ax.set_ylabel("log10 regret")
    twin = ax.twinx()
    twin.plot(sigma, overlap, marker="s", color="#1f4e79", label="top-10 overlap")
    twin.set_ylabel("top-10 overlap")
    twin.set_ylim(0.0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _masked_mean(values, valid):
    masked = np.where(valid[..., None], values, np.nan)
    return np.nanmean(masked, axis=1)


def _true_total(quanta, valid):
    totals = np.sum(quanta, axis=-1)
    totals = np.where(valid, totals, np.nan)
    return np.nanmean(totals, axis=1)


def _coverage(samples, truth, valid, low_q, high_q):
    low = np.quantile(samples, low_q, axis=0)
    high = np.quantile(samples, high_q, axis=0)
    inside = (truth >= low[:, None, :]) & (truth <= high[:, None, :])
    mask = valid[..., None] & np.isfinite(truth)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(inside[mask]))


def _quantile_log_mae(log_samples, truth_log, valid, quanta, probs):
    pred_total = np.sum(np.power(10.0, log_samples), axis=-1)
    true_total = np.sum(quanta, axis=-1)
    errors = []
    for prob in probs:
        pred_q = np.quantile(pred_total, prob, axis=0)
        true_q = np.empty(len(valid), dtype=np.float64)
        for index in range(len(valid)):
            observed = true_total[index, valid[index]]
            true_q[index] = np.quantile(observed, prob) if observed.size else np.nan
        errors.append(np.abs(
            np.log10(np.maximum(pred_q, 1e-30)) - np.log10(np.maximum(true_q, 1e-30))
        ))
    return float(np.nanmean(np.concatenate(errors)))


def _gaussian_nll(samples, truth, valid):
    mu = samples.mean(axis=0)
    var = samples.var(axis=0) + 1e-4
    resid = truth - mu[:, None, :]
    nll = 0.5 * (np.log(var)[:, None, :] + resid ** 2 / var[:, None, :])
    mask = valid[..., None] & np.isfinite(truth)
    per = np.full(len(valid), np.nan)
    for index in range(len(valid)):
        if np.any(mask[index]):
            per[index] = float(np.mean(nll[index][mask[index]]))
    return per


def _by_field(sigma, keep, pred, true, nll, expected, truth_total, samples, truth_log, valid):
    rows = []
    for label in np.unique(np.round(sigma, 8)):
        chosen = keep & (np.round(sigma, 8) == label)
        if not np.any(chosen):
            continue
        rows.append({
            "sigma": float(label),
            "n": int(chosen.sum()),
            "nll": _nanmean(nll[chosen]),
            "log_rmse": _rmse(pred[chosen], true[chosen]),
            "log_mae": _mae(pred[chosen], true[chosen]),
            "coverage_50": _coverage(samples[:, chosen], truth_log[chosen], valid[chosen], 0.25, 0.75),
            "coverage_90": _coverage(samples[:, chosen], truth_log[chosen], valid[chosen], 0.05, 0.95),
            "log_total_mae": _mae(
                np.log10(np.maximum(expected[chosen], 1e-30)),
                np.log10(np.maximum(truth_total[chosen], 1e-30)),
            ),
        })
    return rows


def _subset_metrics(regime, keep, pred, true, expected, truth_total):
    chosen = regime & keep & np.isfinite(truth_total)
    if not np.any(chosen):
        return {"n": 0, "log_rmse": float("nan"), "log_total_mae": float("nan")}
    return {
        "n": int(chosen.sum()),
        "log_rmse": _rmse(pred[chosen], true[chosen]),
        "log_total_mae": _mae(
            np.log10(np.maximum(expected[chosen], 1e-30)),
            np.log10(np.maximum(truth_total[chosen], 1e-30)),
        ),
    }


def _rmse(pred, true):
    if pred.size == 0 or not np.isfinite(true).any():
        return float("nan")
    return float(np.sqrt(np.nanmean((pred - true) ** 2)))


def _mae(pred, true):
    if np.size(pred) == 0:
        return float("nan")
    return float(np.nanmean(np.abs(pred - true)))


def _median_ae(pred, true):
    if np.size(pred) == 0:
        return float("nan")
    return float(np.nanmedian(np.abs(pred - true)))


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _spearman(left, right):
    from scipy.stats import spearmanr

    if len(left) < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    coef, _p = spearmanr(left, right)
    return float(coef)


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
