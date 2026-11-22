"""Shared metrics for the original stochastic MLP and the new surrogates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .layout import log_to_quanta, quanta_to_log


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
    samples = predictor.sample_log(design, sigma, n_samples=int(n_mc), rng=rng, include_mean_uncertainty=True)
    cloud = predictor.sample_log(design, sigma, n_samples=int(n_mc), rng=rng + 1, include_mean_uncertainty=False)
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
    obs_arith = _masked_mean(quanta, valid)
    pred_arith = np.asarray(predictor.arithmetic_mean(design, sigma), dtype=np.float64)
    metrics["arith_log_rmse"] = _rmse(
        np.log10(np.maximum(pred_arith[keep], 1e-30)),
        np.log10(np.maximum(obs_arith[keep], 1e-30)),
    )
    metrics["arith_log_mae"] = _mae(
        np.log10(np.maximum(pred_arith[keep], 1e-30)),
        np.log10(np.maximum(obs_arith[keep], 1e-30)),
    )
    linear_nll = predictor.linear_negative_log_likelihood(design, sigma, quanta, valid)
    metrics["linear_nll"] = _nanmean(np.where(keep, linear_nll, np.nan))
    linear_samples = log_to_quanta(cloud, predictor.log_eps)
    ratio_cv = predictor.ratio_cv(design, sigma) if hasattr(predictor, "ratio_cv") else None
    metrics["shape"] = _shape_report(linear_samples, quanta, valid, sigma, keep, ratio_cv)
    metrics["tail"] = _tail_report(linear_samples, quanta, valid, keep)
    metrics["transition"] = _transition(predictor, design, sigma, keep)
    metrics["arith_by_field"] = _arith_by_field(sigma, keep, pred_arith, obs_arith, linear_nll)
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


def evaluate_shape_model(predictor, table, indices, n_samples=2048, rng=0):
    """Score the ratio curve on held-out designs using their observed means.

    The shape model is not fit on these designs. Zero field is reported and
    then left out of the density, because those seeds sit on the mean.
    """
    from .factorized import MEAN_FLOOR, QUANTILE_LEVELS, _mixture_cdf_numpy

    indices = np.asarray(indices, dtype=int)
    design = table.design[indices]
    sigma = np.asarray(table.sigma[indices], dtype=np.float64)
    quanta = np.asarray(table.quanta[indices], dtype=np.float64)
    valid = np.asarray(table.valid[indices], dtype=bool)
    nll = predictor.working_nll(design, sigma, quanta, valid)
    curve_cv = predictor.curve_cv(design, sigma)
    params = predictor.parameters_at(design, sigma)
    generator = np.random.default_rng(int(rng))
    rows = []
    for label in np.unique(np.round(sigma, 8)):
        chosen = np.flatnonzero(np.round(sigma, 8) == label)
        if chosen.size == 0:
            continue
        if label > 0.0:
            draw = predictor.sample_ratios(
                design[chosen], np.full(chosen.size, label), n_samples=int(n_samples), rng=generator
            )
            low_50, high_50 = np.quantile(draw, [0.25, 0.75], axis=0)
            low_90, high_90 = np.quantile(draw, [0.05, 0.95], axis=0)
        per_mode = [{"obs_cv": [], "obs_frac": [], "cdf_error": [], "cover_50": [], "cover_90": [], "pred_frac": []} for _ in range(3)]
        for local, index in enumerate(chosen):
            ratio, usable = _observed_ratios(quanta[index], valid[index], MEAN_FLOOR)
            for mode in range(3):
                seeds = ratio[usable[:, mode], mode]
                if seeds.size < 2:
                    continue
                bucket = per_mode[mode]
                bucket["obs_cv"].append(float(seeds.std() / max(float(seeds.mean()), 1e-30)))
                bucket["obs_frac"].append(float(np.mean(seeds < 1.0)))
                if label > 0.0:
                    bucket["pred_frac"].append(
                        _ratio_cdf(1.0, params, index, mode, predictor.config.head, _mixture_cdf_numpy)
                    )
                    bucket["cdf_error"].append(
                        _quantile_cdf_error(seeds, params, index, mode, predictor.config.head, QUANTILE_LEVELS, _mixture_cdf_numpy)
                    )
                    bucket["cover_50"].append(float(np.mean((seeds >= low_50[local, mode]) & (seeds <= high_50[local, mode]))))
                    bucket["cover_90"].append(float(np.mean((seeds >= low_90[local, mode]) & (seeds <= high_90[local, mode]))))
        positive = chosen[np.isfinite(nll[chosen])]
        modes = []
        curve = np.median(curve_cv[chosen], axis=0)
        for mode in range(3):
            bucket = per_mode[mode]
            modes.append({
                "mode": mode,
                "obs_cv": _finite_median(bucket["obs_cv"]),
                "curve_cv": float(curve[mode]),
                "obs_frac_below": _finite_median(bucket["obs_frac"]),
                "pred_frac_below": _finite_median(bucket["pred_frac"]) if label > 0.0 else None,
                "quantile_cdf_mae": _finite_median(bucket["cdf_error"]) if label > 0.0 else None,
                "coverage_50": _finite_median(bucket["cover_50"]) if label > 0.0 else None,
                "coverage_90": _finite_median(bucket["cover_90"]) if label > 0.0 else None,
            })
        rows.append({
            "sigma": float(label),
            "n": int(chosen.size),
            "scored": bool(label > 0.0),
            "working_nll": _nanmean(nll[positive]) if label > 0.0 else None,
            "modes": modes,
        })
    scored = np.isfinite(nll)
    return {
        "likelihood": "working_product_of_normalized_ratios",
        "dependence": (
            "Ratios that share a sample mean are dependent. The product density is a "
            "training objective. Coverage and the order-statistic CDF are the calibration checks."
        ),
        "n": int(len(indices)),
        "working_nll": _nanmean(nll[scored]),
        "by_field": rows,
    }


def evaluate_mean_model(predictor, table, indices, ranking_k=10):
    """Score arithmetic-mean heating. The ratio distribution is not used."""
    indices = np.asarray(indices, dtype=int)
    design = table.design[indices]
    sigma = table.sigma[indices]
    quanta = table.quanta[indices]
    valid = table.valid[indices]
    predicted = predictor.arithmetic_mean(design, sigma)
    observed = _masked_mean(quanta, valid)
    keep = valid.any(axis=1) & np.isfinite(observed).all(axis=1) & (observed > 0.0).all(axis=1)
    log_pred = np.log10(np.maximum(predicted, 1e-30))
    log_obs = np.log10(np.maximum(observed, 1e-30))
    return {
        "n": int(keep.sum()),
        "arith_log_rmse": _rmse(log_pred[keep], log_obs[keep]),
        "arith_log_mae": _mae(log_pred[keep], log_obs[keep]),
        "by_field": _arith_by_field(sigma, keep, predicted, observed, np.full(len(sigma), np.nan)),
        "ranking": rank_metrics(sigma[keep], predicted[keep].sum(axis=-1), _true_total(quanta, valid)[keep], k=ranking_k),
    }


def _observed_ratios(quanta, valid, mean_floor):
    quanta = np.asarray(quanta, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    safe = np.where(valid[:, None], quanta, 0.0)
    count = np.maximum(valid.astype(np.float64).sum(), 1.0)
    observed = safe.sum(axis=0) / count
    usable = valid[:, None] & (observed[None, :] >= float(mean_floor))
    ratio = np.ones(safe.shape, dtype=np.float64)
    np.divide(safe, observed[None, :], out=ratio, where=usable)
    return ratio, usable


def _quantile_cdf_error(seeds, params, index, mode, head, levels, mixture_cdf):
    errors = []
    for level in levels:
        quantile = float(np.quantile(seeds, level))
        errors.append(abs(_ratio_cdf(quantile, params, index, mode, head, mixture_cdf) - level))
    return float(np.mean(errors))


def _ratio_cdf(ratio, params, index, mode, head, mixture_cdf):
    if head == "gamma":
        from scipy.special import gammainc

        shape = float(params["shape"][index, mode])
        return float(gammainc(shape, shape * max(ratio, 0.0)))
    weight = float(1.0 / (1.0 + np.exp(-params["logit"][index, mode])))
    cdf = mixture_cdf(
        np.asarray(ratio, dtype=np.float64),
        np.asarray(params["gaussian_scale"][index, mode], dtype=np.float64),
        np.asarray(params["lognormal_scale"][index, mode], dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
    )
    return float(np.asarray(cdf).reshape(-1)[0])


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


def _cloud_stats(cloud):
    stats = []
    for mode in range(cloud.shape[-1]):
        values = np.asarray(cloud[..., mode], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if (
            values.size < 2
            or not np.isfinite(values.mean())
            or values.mean() <= 0.0
            or values.std() <= 1e-8 * max(abs(float(values.mean())), 1.0)
        ):
            stats.append((float("nan"), float("nan"), float("nan")))
            continue
        centered = (values - values.mean()) / values.std()
        stats.append((
            float(values.std() / values.mean()),
            float(np.mean(values < values.mean())),
            float(np.mean(centered ** 3)),
        ))
    return stats


def _shape_report(samples, quanta, valid, sigma, keep, ratio_cv=None):
    """Predicted and observed coefficient of variation, skew, and mass below the mean."""
    rows = []
    for label in np.unique(np.round(sigma, 8)):
        chosen = np.flatnonzero(keep & (np.round(sigma, 8) == label))
        if chosen.size == 0:
            continue
        observed = [[] for _ in range(3)]
        predicted = [[] for _ in range(3)]
        for index in chosen:
            for mode, (cv, frac, skew) in enumerate(_cloud_stats(quanta[index, valid[index]])):
                observed[mode].append((cv, frac, skew))
            for mode, (cv, frac, skew) in enumerate(_cloud_stats(samples[:, index, :])):
                predicted[mode].append((cv, frac, skew))
        entry = {"sigma": float(label), "n": int(chosen.size), "modes": []}
        for mode in range(3):
            obs = np.asarray(observed[mode], dtype=np.float64)
            pred = np.asarray(predicted[mode], dtype=np.float64)
            entry["modes"].append({
                "mode": mode,
                "obs_cv": _finite_median(obs[:, 0]) if obs.size else float("nan"),
                "pred_cv": _finite_median(pred[:, 0]) if pred.size else float("nan"),
                "obs_frac_below": _finite_median(obs[:, 1]) if obs.size else float("nan"),
                "pred_frac_below": _finite_median(pred[:, 1]) if pred.size else float("nan"),
                "obs_skew": _finite_median(obs[:, 2]) if obs.size else float("nan"),
                "pred_skew": _finite_median(pred[:, 2]) if pred.size else float("nan"),
                "ratio_cv": _finite_median(np.asarray(ratio_cv)[chosen, mode]) if ratio_cv is not None else None,
            })
        rows.append(entry)
    return rows


def _tail_report(samples, quanta, valid, keep):
    """Log error of the 0.9 total quantile and of CVaR at 0.9."""
    pred_total = np.sum(samples, axis=-1)
    pred_q = np.quantile(pred_total, 0.9, axis=0)
    level = pred_q
    tail = pred_total >= level[None, :]
    pred_cvar = np.sum(np.where(tail, pred_total, 0.0), axis=0) / np.maximum(tail.sum(axis=0), 1)
    true_q = np.full(len(valid), np.nan)
    true_cvar = np.full(len(valid), np.nan)
    for index in np.flatnonzero(keep):
        observed = np.sum(quanta[index, valid[index]], axis=-1)
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            continue
        cutoff = np.quantile(observed, 0.9)
        true_q[index] = cutoff
        upper = observed[observed >= cutoff]
        true_cvar[index] = float(upper.mean()) if upper.size else float("nan")
    chosen = keep & np.isfinite(true_q) & np.isfinite(pred_q)
    return {
        "q90_log_mae": _mae(np.log10(np.maximum(pred_q[chosen], 1e-30)), np.log10(np.maximum(true_q[chosen], 1e-30))),
        "cvar90_log_mae": _mae(
            np.log10(np.maximum(pred_cvar[chosen], 1e-30)),
            np.log10(np.maximum(true_cvar[chosen], 1e-30)),
        ),
    }


def _transition(predictor, design, sigma, keep):
    if not hasattr(predictor, "shape_parameters"):
        return None
    shape = predictor.shape_parameters(design, sigma)
    if shape is None:
        return None
    rows = []
    for label in np.unique(np.round(sigma, 8)):
        chosen = keep & (np.round(sigma, 8) == label)
        if not np.any(chosen):
            continue
        row = {"sigma": float(label)}
        for key, value in shape.items():
            row[key] = np.median(np.asarray(value)[chosen], axis=0).tolist()
        rows.append(row)
    return rows


def _arith_by_field(sigma, keep, pred, obs, nll):
    rows = []
    for label in np.unique(np.round(sigma, 8)):
        chosen = keep & (np.round(sigma, 8) == label)
        if not np.any(chosen):
            continue
        rows.append({
            "sigma": float(label),
            "arith_log_rmse": _rmse(np.log10(np.maximum(pred[chosen], 1e-30)), np.log10(np.maximum(obs[chosen], 1e-30))),
            "linear_nll": _nanmean(nll[chosen]),
        })
    return rows


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


def _finite_median(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmedian(values))


def _spearman(left, right):
    from scipy.stats import spearmanr

    if len(left) < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    coef, _p = spearmanr(left, right)
    return float(coef)


def plot_factorized_board(out, data=None, holdout_fields=(0.5, 2.0), seed=0):
    """Comparison figures for a finished factorized benchmark directory."""
    import json
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out = Path(out)
    reports = {}
    for path in sorted(out.glob("*/test.json")):
        name = path.parent.name
        reports[name] = {
            "test": json.loads(path.read_text()),
            "interp": json.loads((path.parent / "interp.json").read_text()) if (path.parent / "interp.json").is_file() else None,
        }
    if not reports:
        return
    _plot_arith_and_rank(reports, out, plt)
    _plot_transition(reports, out, plt)
    _plot_cv(reports, out, plt)
    if data is not None:
        _plot_mode_distributions(reports, out, data, holdout_fields, seed, plt)


def _plot_arith_and_rank(reports, out, plt):
    names = [name for name in _BOARD_MODELS if name in reports]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, split, title in (
        (axes[0], "test", "unseen designs, trained fields"),
        (axes[1], "interp", "seen designs, held-out fields"),
    ):
        for name in names:
            rows = (reports[name].get(split) or {}).get("arith_by_field")
            if not rows:
                continue
            ax.plot(
                [row["sigma"] for row in rows],
                [row["arith_log_rmse"] for row in rows],
                marker="o",
                ms=4,
                color=_BOARD_COLORS[name],
                label=name,
            )
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("stray field (V/m)")
        ax.set_title(title)
        ax.set_ylabel("log10 RMSE of arithmetic mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "arith_error_by_field.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for name in names:
        rows = reports[name]["test"].get("ranking") or []
        if not rows:
            continue
        axes[0].plot(
            [row["sigma"] for row in rows],
            [row["spearman_log"] for row in rows],
            marker="o",
            ms=4,
            color=_BOARD_COLORS[name],
            label=name,
        )
        axes[1].plot(
            [row["sigma"] for row in rows],
            [row["log10_regret"] for row in rows],
            marker="o",
            ms=4,
            color=_BOARD_COLORS[name],
            label=name,
        )
    axes[0].set_ylabel("Spearman of log10 total heating")
    axes[1].set_ylabel("log10 regret of the selected design")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("stray field (V/m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ranking_by_field.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(names))
    cover_50 = [reports[name]["test"].get("coverage_50") or 0.0 for name in names]
    cover_90 = [reports[name]["test"].get("coverage_90") or 0.0 for name in names]
    ax.bar(x - 0.16, cover_50, width=0.32, label="50% interval", color="#1f4e79")
    ax.bar(x + 0.16, cover_90, width=0.32, label="90% interval", color="#7aa6c2")
    ax.axhline(0.50, color="#1f4e79", lw=0.8, ls="--")
    ax.axhline(0.90, color="#7aa6c2", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("coverage on held-out designs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=140)
    plt.close(fig)


def _plot_transition(reports, out, plt):
    names = [name for name in ("mlp_gauss_lognorm", "transformer_gauss_lognorm", "transformer_gauss_lognorm_l0") if name in reports]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=True)
    mode_style = ("-", "--", ":")
    for name in names:
        rows = reports[name]["test"].get("transition") or []
        rows = [row for row in rows if "gaussian_scale" in row]
        if not rows:
            continue
        sigma = [row["sigma"] for row in rows]
        for mode in range(3):
            label = name if mode == 0 else None
            axes[0].plot(sigma, [row["gaussian_scale"][mode] for row in rows], mode_style[mode], color=_BOARD_COLORS[name], label=label)
            axes[1].plot(sigma, [row["lognormal_scale"][mode] for row in rows], mode_style[mode], color=_BOARD_COLORS[name], label=label)
            axes[2].plot(sigma, [row["lognormal_weight"][mode] for row in rows], mode_style[mode], color=_BOARD_COLORS[name], label=label)
    axes[0].set_ylabel("Gaussian scale")
    axes[1].set_ylabel("lognormal scale")
    axes[2].set_ylabel("lognormal weight")
    axes[2].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("stray field (V/m)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("learned ratio parameters (solid tangent, dashed normal, dotted vertical)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "transition.png", dpi=140)
    plt.close(fig)


def _mode_cv(mode):
    if mode.get("ratio_cv") is not None:
        return mode["ratio_cv"]
    return mode["pred_cv"]


def _plot_cv(reports, out, plt):
    target = reports.get("transformer_gauss_lognorm")
    if target is None:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    rows = [row for row in target["test"]["shape"] if row["sigma"] > 0.0]
    ax.plot([row["sigma"] for row in rows], [row["modes"][0]["obs_cv"] for row in rows], color="#222222", marker="o", label="observed tangent")
    for name, mode_label in (
        ("transformer_gauss_lognorm", "transformer Gauss–lognormal"),
        ("mlp_gauss_lognorm", "MLP Gauss–lognormal"),
        ("transformer_mixture", "transformer log-mixture"),
    ):
        block = reports.get(name)
        if block is None:
            continue
        series = [row for row in block["test"]["shape"] if row["sigma"] > 0.0]
        ax.plot(
            [row["sigma"] for row in series],
            [_mode_cv(row["modes"][0]) for row in series],
            marker="o",
            color=_BOARD_COLORS[name],
            label=mode_label,
        )
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("stray field (V/m)")
    ax.set_ylabel("coefficient of variation")
    ax.set_title("tangent mode, held-out designs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cv_by_field.png", dpi=140)
    plt.close(fig)


def _plot_mode_distributions(reports, out, data, holdout_fields, seed, plt):
    from scipy.special import ndtr

    from .data import load_field_data, make_split

    learned = (reports.get("transformer_gauss_lognorm") or {}).get("test", {}).get("transition")
    if not learned:
        return
    table = load_field_data(data)
    split = make_split(table, holdout_fields=holdout_fields, seed=seed)
    fields = (0.1, 1.0, 10.0)
    mode_names = ("tangent", "normal", "vertical")
    fig, axes = plt.subplots(len(fields), 3, figsize=(10.2, 7.6), sharex=True)
    grid = np.logspace(-2.0, 1.2, 500)
    bins = np.logspace(-2.0, 1.2, 28)
    for row, sigma in enumerate(fields):
        params = min(learned, key=lambda item: abs(item["sigma"] - sigma))
        chosen = [int(index) for index in split.test if abs(float(table.sigma[index]) - sigma) < 1e-8]
        for mode in range(3):
            ratios = []
            for index in chosen:
                values = table.quanta[index, table.valid[index], mode]
                values = values[np.isfinite(values) & (values > 0.0)]
                if values.size < 2 or values.mean() <= 0.0:
                    continue
                ratios.append(values / values.mean())
            cloud = np.concatenate(ratios) if ratios else np.zeros(0)
            ax = axes[row, mode]
            if cloud.size:
                ax.hist(cloud, bins=bins, density=True, color="#c5d4e4", label="observed n / mean")
            pdf = _mixture_pdf(
                grid,
                params["gaussian_scale"][mode],
                params["lognormal_scale"][mode],
                params["lognormal_weight"][mode],
                ndtr,
            )
            ax.plot(grid, pdf, color="#1f4e79", lw=1.6, label="learned mixture")
            ax.set_xscale("log")
            ax.set_xlim(1e-2, 10.0 ** 1.2)
            if row == len(fields) - 1:
                ax.set_xlabel("phonon / design mean")
            if row == 0:
                ax.set_title(mode_names[mode])
            if mode == 0:
                ax.set_ylabel("{} V/m".format(sigma))
            if row == 0 and mode == 2:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("within-design ratio and the median learned mixture", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "mode_distributions.png", dpi=140)
    plt.close(fig)


def _mixture_pdf(ratio, gaussian_scale, lognormal_scale, weight, ndtr):
    z = 1.0 / gaussian_scale
    mass = float(ndtr(z))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    mean_x = 1.0 + gaussian_scale * (phi / max(mass, 1e-30))
    x = ratio * mean_x
    gauss = np.exp(-0.5 * ((x - 1.0) / gaussian_scale) ** 2) / (
        gaussian_scale * math.sqrt(2.0 * math.pi) * max(mass, 1e-30)
    ) * mean_x
    mu = -0.5 * lognormal_scale ** 2
    safe = np.maximum(ratio, 1e-300)
    lognorm = np.exp(-0.5 * ((np.log(safe) - mu) / lognormal_scale) ** 2) / (
        safe * lognormal_scale * math.sqrt(2.0 * math.pi)
    )
    return np.where(ratio > 0.0, (1.0 - weight) * gauss + weight * lognorm, 0.0)


_BOARD_MODELS = (
    "mlp_mixture",
    "transformer_mixture",
    "mlp_gauss_lognorm",
    "transformer_gauss_lognorm",
    "transformer_gauss_lognorm_l0",
    "transformer_gauss_lognorm_copula",
    "transformer_gamma",
    "legacy_stochastic",
)

_BOARD_COLORS = {
    "mlp_mixture": "#8d6a9f",
    "transformer_mixture": "#5c4d7a",
    "mlp_gauss_lognorm": "#d4762c",
    "transformer_gauss_lognorm": "#1f4e79",
    "transformer_gauss_lognorm_l0": "#3d8b8b",
    "transformer_gauss_lognorm_copula": "#1a7a3a",
    "transformer_gamma": "#a33b3b",
    "legacy_stochastic": "#666666",
}


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
