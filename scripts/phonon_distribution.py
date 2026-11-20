"""Shape of the end-phonon seeds on the stray-field sweep.

Each positive amplitude stores 20 final tangent-mode quanta per design. This
compares those seeds with a Gaussian, an exponential (classical Boltzmann
energy of one harmonic mode), a gamma, and a lognormal. A Poisson with the
same mean is far too narrow to be a candidate: its coefficient of variation
is 1/sqrt(mean).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplcache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "field_s2um"
OUT = ROOT / "results" / "phonon_distribution"
FIELDS = (0.1, 1.0, 5.0, 10.0)
MODE = 0
BIN_EDGE = 4.0
BIN_WIDTH = 0.2
FAMILIES = ("gaussian", "exponential", "gamma", "lognormal")


def load_field(amplitude):
    tag = "{:.3g}".format(float(amplitude)).replace(".", "p").replace("-", "m")
    path = SWEEP / "noise_{}".format(tag) / "joint.npz"
    archive = np.load(path)
    stored = float(np.asarray(archive["noise_std_Vm"]).reshape(-1)[0])
    if abs(stored - float(amplitude)) > 1e-8:
        raise RuntimeError("{} records {} V/m".format(path, stored))
    quanta = np.asarray(archive["quanta"], dtype=np.float64)
    valid = np.isfinite(quanta).all(axis=-1) & (quanta >= 0.0).all(axis=-1)
    return quanta[:, :, MODE], valid


def fit_scores(samples):
    scores = {}
    spread = float(samples.std(ddof=1))
    if spread > 0.0:
        logp = stats.norm.logpdf(samples, float(samples.mean()), spread)
        scores["gaussian"] = 4.0 - 2.0 * float(np.sum(logp))
    for name, dist, k in (
        ("exponential", stats.expon, 1),
        ("gamma", stats.gamma, 2),
        ("lognormal", stats.lognorm, 2),
    ):
        try:
            args = dist.fit(samples, floc=0)
            logp = dist.logpdf(samples, *args)
        except Exception:
            continue
        if np.isfinite(logp).all():
            scores[name] = 2.0 * k - 2.0 * float(np.sum(logp))
    shape = np.nan
    if "gamma" in scores:
        try:
            shape = float(stats.gamma.fit(samples, floc=0)[0])
        except Exception:
            shape = np.nan
    return scores, shape


def analyze_field(amplitude):
    quanta, valid = load_field(amplitude)
    ratios = []
    below = []
    share_above = []
    ratio_median = []
    cvs = []
    skews = []
    wins = {name: 0 for name in FAMILIES}
    gaps = {"lognormal_minus_gaussian": [], "lognormal_minus_exponential": [], "lognormal_minus_gamma": []}
    shapes = []
    means = []
    used = 0
    for index in range(quanta.shape[0]):
        samples = quanta[index, valid[index]]
        samples = samples[np.isfinite(samples) & (samples > 0.0)]
        if samples.size < 8:
            continue
        mean = float(samples.mean())
        ratios.append(samples / mean)
        below.append(float(np.mean(samples < mean)))
        above = samples[samples > mean]
        share_above.append(float(above.sum() / samples.sum()))
        ratio_median.append(float(np.median(samples) / mean))
        cvs.append(float(samples.std(ddof=0) / mean))
        skews.append(float(stats.skew(samples)))
        means.append(mean)
        scores, shape = fit_scores(samples)
        if len(scores) < len(FAMILIES):
            continue
        winner = min(scores, key=scores.get)
        wins[winner] += 1
        used += 1
        shapes.append(shape)
        gaps["lognormal_minus_gaussian"].append(scores["lognormal"] - scores["gaussian"])
        gaps["lognormal_minus_exponential"].append(scores["lognormal"] - scores["exponential"])
        gaps["lognormal_minus_gamma"].append(scores["lognormal"] - scores["gamma"])
    pooled = np.concatenate(ratios)
    edges = np.linspace(0.0, BIN_EDGE, int(round(BIN_EDGE / BIN_WIDTH)) + 1)
    counts, _edges = np.histogram(pooled, bins=edges)
    density = counts / (pooled.size * BIN_WIDTH)
    return {
        "sigma_Vm": float(amplitude),
        "mode": "tangent",
        "n_designs": int(used),
        "n_seeds": int(pooled.size),
        "median_cv": _median(cvs),
        "median_skew": _median(skews),
        "median_fraction_below_mean": _median(below),
        "mean_fraction_below_mean": float(np.mean(below)),
        "median_share_above_mean": _median(share_above),
        "median_seed_over_mean": _median(ratio_median),
        "overflow_above_4": float(np.mean(pooled > BIN_EDGE)),
        "median_gamma_shape": _median(shapes),
        "poisson_cv_at_median_mean": float(1.0 / np.sqrt(np.median(means))),
        "aic_wins": wins,
        "aic_win_fraction": {name: wins[name] / used for name in FAMILIES},
        "median_aic_gap": {name: _median(values) for name, values in gaps.items()},
        "histogram": {
            "edges": edges.tolist(),
            "density": density.tolist(),
        },
    }


def _median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values))


def _bin_density(edges, cdf):
    width = float(edges[1] - edges[0])
    return np.asarray([(cdf(edges[i + 1]) - cdf(edges[i])) / width for i in range(len(edges) - 1)])


def plot(summary):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), sharex=True)
    for ax, row in zip(axes.ravel(), summary["fields"]):
        edges = np.asarray(row["histogram"]["edges"])
        density = np.asarray(row["histogram"]["density"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        exponential = _bin_density(edges, lambda x: 1.0 - np.exp(-np.maximum(x, 0.0)))
        sd = max(row["median_cv"], 1e-6)
        gaussian = _bin_density(edges, lambda x: stats.norm.cdf(x, loc=1.0, scale=sd))
        ax.bar(centers, density, width=BIN_WIDTH * 0.92, color="#e07a3d", label="seeds")
        ax.plot(centers, exponential, color="#2f6fdb", lw=1.6, label="exponential")
        ax.plot(centers, gaussian, color="#7a4ea3", lw=1.6, label="gaussian")
        ax.set_title("{:g} V/m".format(row["sigma_Vm"]))
        ax.text(
            0.98,
            0.95,
            "CV {:.3g}\n{:.0%} below mean".format(row["median_cv"], row["median_fraction_below_mean"]),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.set_xlim(0.0, BIN_EDGE)
        ax.set_ylabel("density")
    axes[1, 0].set_xlabel("quanta / design mean")
    axes[1, 1].set_xlabel("quanta / design mean")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "within_design.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = {
        "source": "results/field_s2um",
        "heating": "end",
        "comparison": (
            "Within each design the tangent-mode seeds are divided by that design's mean. "
            "AIC uses a Gaussian, an exponential fixed at zero, a gamma fixed at zero, and a lognormal fixed at zero."
        ),
        "fields": [analyze_field(amplitude) for amplitude in FIELDS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(summary))
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot(summary)
    for row in summary["fields"]:
        wins = ", ".join("{} {:.0%}".format(name, row["aic_win_fraction"][name]) for name in FAMILIES)
        print(
            "{:g} V/m  CV {:.3f}  below {:.0%}  seed/mean {:.2f}  {}".format(
                row["sigma_Vm"],
                row["median_cv"],
                row["median_fraction_below_mean"],
                row["median_seed_over_mean"],
                wins,
            )
        )


if __name__ == "__main__":
    main()
