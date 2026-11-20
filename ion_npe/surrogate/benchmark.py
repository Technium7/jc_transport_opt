"""Train and compare the residual MLP, the structured transformer, and the original network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import describe, load_field_data, make_split, table_fingerprint
from .evaluate import (
    evaluate_legacy,
    evaluate_predictor,
    fit_legacy,
    plot_coverage,
    plot_distributions,
    plot_field_errors,
    plot_prediction,
    plot_ranking,
    plot_training,
    public_metrics,
    save_metrics,
)
from .model import EnsemblePredictor, SurrogateConfig, SurrogatePredictor, default_device
from .train import save_training, train_surrogate


COMPARISONS = (
    ("mlp_gaussian", "mlp", 1),
    ("mlp_mixture", "mlp", 3),
    ("transformer_gaussian", "transformer", 1),
    ("transformer_mixture", "transformer", 3),
)


def run_benchmark(
    data,
    out,
    epochs=30,
    patience=8,
    batch_size=128,
    holdout_fields=(0.5, 2.0),
    seed=0,
    device=None,
    legacy_epochs=12,
    ensemble=0,
    n_mc=96,
):
    """Fit the comparison models on one split and write metrics and plots."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = default_device()
    table = load_field_data(data)
    split = make_split(table, holdout_fields=holdout_fields, seed=seed)
    print(describe(table), flush=True)
    print(
        "split train {} val {} test {} interp {}  holdout {}".format(
            len(split.train), len(split.val), len(split.test), len(split.interp), list(holdout_fields)
        ),
        flush=True,
    )
    summaries = []
    field_rows = []
    coverage_rows = []
    for name, arch, components in COMPARISONS:
        print("training {}".format(name), flush=True)
        config = SurrogateConfig(arch=arch, n_components=components)
        predictor = train_surrogate(
            split,
            config,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            device=device,
            log_every=5,
        )
        path = out / "{}.pt".format(name)
        save_training(path, predictor, split, predictor.history, epoch=len(predictor.history["train"]), best_score=predictor.best_score)
        metrics = evaluate_predictor(predictor, table, split.test, n_mc=n_mc, rng=seed)
        interp = None
        if len(split.interp):
            interp = evaluate_predictor(predictor, table, split.interp, n_mc=n_mc, rng=seed + 1)
        _write_model_report(out / name, predictor, metrics, interp)
        summaries.append({"name": name, "test": public_metrics(metrics), "interp": None if interp is None else public_metrics(interp)})
        field_rows.append((name, metrics["by_field"]))
        coverage_rows.append((name, metrics))
        if name == "transformer_mixture":
            _try_plot(lambda: _example_plots(predictor, table, split, out / name / "distributions.png", seed))
    print("training original stochastic MLP", flush=True)
    legacy = fit_legacy(split, epochs=legacy_epochs, seed=seed)
    if legacy is not None:
        legacy_metrics = evaluate_legacy(legacy, table, split.test, n_mc=min(n_mc, 32), rng=seed)
        save_metrics(out / "legacy_stochastic" / "test.json", legacy_metrics)
        _try_plot(lambda: plot_prediction(legacy_metrics, out / "legacy_stochastic" / "pred_vs_sim.png", title="original stochastic MLP"))
        summaries.append({"name": "legacy_stochastic", "test": public_metrics(legacy_metrics), "interp": None})
        field_rows.append(("legacy_stochastic", legacy_metrics["by_field"]))
        coverage_rows.append(("legacy_stochastic", legacy_metrics))
    if int(ensemble) > 1:
        print("training ensemble of {}".format(ensemble), flush=True)
        members = []
        for index in range(int(ensemble)):
            member = train_surrogate(
                split,
                SurrogateConfig(arch="transformer", n_components=3),
                epochs=max(int(epochs) // 2, 8),
                batch_size=batch_size,
                patience=max(int(patience) // 2, 4),
                seed=seed + 100 + index,
                device=device,
                log_every=5,
            )
            members.append(member)
        group = EnsemblePredictor(members, device=device)
        from .data import split_payload
        group.save(out / "transformer_mixture_ensemble", extra={"split": split_payload(split)})
        ens_metrics = evaluate_predictor(group, table, split.test, n_mc=n_mc, rng=seed)
        save_metrics(out / "transformer_mixture_ensemble" / "test.json", ens_metrics)
        summaries.append({
            "name": "transformer_mixture_ensemble",
            "test": public_metrics(ens_metrics),
            "epistemic_mean": float(np.nanmean(group.epistemic_total_std(table.design[split.test], table.sigma[split.test]))),
        })
        field_rows.append(("ensemble", ens_metrics["by_field"]))
        coverage_rows.append(("ensemble", ens_metrics))
    _try_plot(lambda: plot_field_errors(field_rows, out / "error_by_field.png"))
    _try_plot(lambda: plot_coverage(coverage_rows, out / "coverage.png"))
    (out / "comparison.json").write_text(json.dumps(summaries, indent=2))
    (out / "split.json").write_text(json.dumps({
        "fingerprint": table_fingerprint(table),
        "holdout_fields": [float(v) for v in holdout_fields],
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "n_interp": int(len(split.interp)),
        "dropped_invalid": int(table.info.get("n_dropped_invalid", 0)),
    }, indent=2))
    return summaries


def load_saved(path, device=None):
    path = Path(path)
    if path.is_dir():
        return EnsemblePredictor.load(path, device=device)
    return SurrogatePredictor.load(path, device=device)


def saved_split(path):
    path = Path(path)
    weight = path / "member_0.pt" if path.is_dir() else path
    payload = torch.load(weight, map_location="cpu", weights_only=False)
    if "split" not in payload:
        raise ValueError("{} has no saved split".format(path))
    return payload["split"], payload.get("history")


def _write_model_report(directory, predictor, metrics, interp):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_metrics(directory / "test.json", metrics)
    if interp is not None:
        save_metrics(directory / "interp.json", interp)
    _try_plot(lambda: plot_training(predictor.history, directory / "loss.png"))
    _try_plot(lambda: plot_prediction(metrics, directory / "pred_vs_sim.png"))
    _try_plot(lambda: plot_ranking(metrics, directory / "ranking.png"))
    if interp is not None:
        _try_plot(lambda: plot_prediction(interp, directory / "interp_pred_vs_sim.png", title="held-out field amplitudes"))


def _try_plot(draw):
    try:
        draw()
    except Exception as err:
        print("  plot skipped: {}".format(err), flush=True)


def _example_plots(predictor, table, split, path, seed):
    chosen = []
    for sigma, pool in ((0.1, split.test), (10.0, split.test), (0.5, split.interp)):
        match = [int(index) for index in pool if abs(float(table.sigma[index]) - sigma) < 1e-8]
        if match:
            chosen.append(match[len(match) // 2])
    if chosen:
        plot_distributions(predictor, table, chosen[:3], path, rng=seed)
