"""Train the field surrogate, then search at one stray-field amplitude.

Each round fits the transformer Gaussian-lognormal model on the simulations
collected so far, screens a cross-entropy cloud, and integrates the designs
it ranks coldest. The initial fit starts from the checkpoint trained on the
per-field archive.
"""

from pathlib import Path

import numpy as np

from ion_npe.design import DesignSpace
from ion_npe.joint import (
    _append,
    _bounds,
    _draw,
    _simulate_ordered,
    _std,
    _unpack,
)
from ion_npe.plotting import plot_best_design, plot_trajectory
from ion_npe.simulate import simulation_settings, total_heating
from ion_npe.surrogate.data import load_field_data, make_split
from ion_npe.surrogate.model import SurrogateConfig
from ion_npe.surrogate.predict import rank_candidates
from ion_npe.surrogate.train import save_training, train_surrogate


def optimize_at_field(
    out,
    data,
    sigma=10.0,
    init_weights=None,
    iterations=4,
    n_screen=1000,
    n_simulate=40,
    epochs=40,
    patience=10,
    seed=0,
    simulate_kwargs=None,
):
    """Fit on ``data``, then run ``iterations`` of screening and simulation."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    simulate_kwargs = simulation_settings(dict(simulate_kwargs or {}))
    simulate_kwargs["noise_std_Vm"] = float(sigma)
    simulate_kwargs = simulation_settings(simulate_kwargs)
    source = Path(data)
    archive = np.load(source)
    rows = np.asarray(archive["rows"], dtype=float)
    quanta = np.asarray(archive["quanta"], dtype=float)
    totals = np.asarray(archive["totals"], dtype=float) if "totals" in archive.files else np.array([
        float(total_heating(item)) if np.isfinite(item).all() else np.nan for item in quanta
    ])
    origin = np.array(["train"] * len(rows), dtype="U16")
    space = DesignSpace()
    lower, upper = _bounds(space)
    rng = np.random.default_rng(seed)
    local_std = _std(space, lower, upper, 0.03)
    weights = init_weights
    print(
        "field CEM at {:.4g} V/m  {} stored designs  {} seeds".format(
            sigma, len(rows), quanta.shape[1]
        ),
        flush=True,
    )
    for step in range(int(iterations) + 1):
        table_path = out / "round_{}_train.npz".format(step)
        _write_archive(table_path, rows, quanta, totals, sigma, simulate_kwargs["seeds"])
        table = load_field_data(table_path)
        split = make_split(table, holdout_fields=(), seed=seed + step)
        config = _config(weights)
        print(
            "training round {}  {} pairs  init {}".format(
                step, len(table), "checkpoint" if weights else "random"
            ),
            flush=True,
        )
        predictor = train_surrogate(
            split,
            config,
            epochs=epochs if step == 0 else max(int(epochs) // 2, 12),
            patience=patience if step == 0 else max(int(patience) // 2, 4),
            seed=seed + step,
            log_every=5,
            init_weights=weights,
            shape_epochs=max(int(epochs) // 3, 8) if step == 0 else 8,
        )
        weights = out / "round_{}.pt".format(step)
        save_training(
            weights, predictor, split, predictor.history,
            epoch=len(predictor.history["train"]), best_score=predictor.best_score,
        )
        if step == int(iterations):
            break
        finite = np.isfinite(totals)
        order = np.argsort(np.where(finite, totals, np.inf))
        elite = order[: max(16, 2)]
        mean = rows[elite].mean(axis=0)
        std = np.maximum(rows[elite].std(axis=0), 0.5 * local_std)
        best = rows[int(order[0])]
        n_each = max(int(n_simulate) // 2, 1)
        print(
            "CEM round {}  screen {}  simulate {}".format(step + 1, n_screen, n_simulate),
            flush=True,
        )
        elite_pool = _draw(rng, max(int(n_screen) // 2, n_each), mean, std, lower, upper, space)
        best_pool = _draw(rng, max(int(n_screen) // 2, n_each), best, local_std, lower, upper, space)
        for pool, tag, take in (
            (elite_pool, "cem-elite", n_each),
            (best_pool, "cem-best", n_each),
        ):
            rank, scores, _ok = rank_candidates(predictor, pool, float(sigma), rng=seed + step)
            kept, new_q, new_total = _simulate_ordered(
                space, pool, rank, scores, rows, lower, upper, take, simulate_kwargs, tag,
            )
            rows, quanta, totals, origin = _append(
                rows, quanta, totals, origin, kept, new_q, new_total, tag,
            )
        best_now = float(np.nanmin(totals))
        print("  best so far {:.6g} over {} simulations".format(best_now, len(totals)), flush=True)
        _write_archive(out / "joint.npz", rows, quanta, totals, sigma, simulate_kwargs["seeds"], origin)

    finite = np.isfinite(totals)
    plot_i = int(np.argmin(np.where(finite, totals, np.inf)))
    design = _unpack(space, rows[plot_i])
    plot_trajectory(design, None, out / "best_path.png")
    plot_best_design(
        design, out / "best_design.png",
        reference=space.nominal_design(),
        total_quanta=float(totals[plot_i]),
        title="best design at {:.4g} V/m, {:.3g} quanta".format(sigma, float(totals[plot_i])),
    )
    _write_archive(out / "joint.npz", rows, quanta, totals, sigma, simulate_kwargs["seeds"], origin)
    print(
        "saved {}  best {:.6g} quanta  origin {}".format(out, float(totals[plot_i]), origin[plot_i]),
        flush=True,
    )


def _config(path):
    if not path:
        return SurrogateConfig(arch="transformer", head="gauss_lognorm", n_components=1, lambda_mean=1.0)
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {item.name for item in __import__("dataclasses").fields(SurrogateConfig)}
    values = {key: value for key, value in payload["config"].items() if key in allowed}
    return SurrogateConfig(**values)


def _write_archive(path, rows, quanta, totals, sigma, seeds, origin=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        rows=np.asarray(rows, dtype=float),
        quanta=np.asarray(quanta, dtype=float),
        totals=np.asarray(totals, dtype=float),
        noise_std_Vm=np.float64(sigma),
        seeds=np.asarray(seeds, dtype=int),
        n_train=np.int64(len(rows)),
    )
    if origin is not None:
        payload["origin"] = np.asarray(origin, dtype="U16")
    np.savez(path, **payload)
