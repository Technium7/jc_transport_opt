"""Collect near the nominal path, then actively add low-phonon, uncertain designs.

The first sample is a small box around the nominal junction and trajectory.
A stochastic network is fit with Adam and then L-BFGS. Several rounds then
draw candidates from a Gaussian centered on that same nominal design, simulate
the ones predicted to have low heating and high uncertainty, and retrain.
"""

from pathlib import Path

import numpy as np

from .active import acquire
from .constants import CROSS_CAP_M, HEIGHT_CAP_M, L_SCALE, SIM_DT
from .dataset import Dataset, collect_dataset, design_is_admissible
from .design import DesignSpace
from .plotting import plot_active, plot_losses, plot_prediction, plot_trajectory
from .simulate import simulate_design, total_heating
from .train import gaussian_nll, lbfgs_polish, log_rmse, train_predictor, to_log
from .trajectory import path_geometry


def _prepend(dataset, vector, quanta, length_m, duration_s):
    quanta = np.asarray(quanta, dtype=float)
    if quanta.ndim == 2:
        quanta = quanta[None, :, :]
    ideal = np.full_like(quanta, np.nan)
    return Dataset(
        np.vstack([np.asarray(vector, dtype=float).reshape(1, -1), dataset.x]),
        np.concatenate([quanta, dataset.quanta], axis=0),
        np.concatenate([ideal, dataset.quanta_ideal], axis=0),
        np.concatenate([[length_m], dataset.length_m]),
        np.concatenate([[duration_s], dataset.duration_s]),
        dataset.seeds,
        dataset.space,
    )


def _split(n, test_fraction, seed):
    """Keep index 0, the nominal design, in the training rows."""
    rng = np.random.default_rng(seed)
    rest = rng.permutation(np.arange(1, n))
    n_test = int(round(test_fraction * (n - 1)))
    n_test = min(max(n_test, 1), n - 2)
    return np.concatenate([[0], rest[n_test:]]), rest[:n_test]


def _coverage(predictor, x, quanta, seed):
    """Fraction of designs whose mean total lies inside mean ± one std."""
    pred = predictor.predict_heating(x, n_samples=48, rng=np.random.default_rng(seed))
    truth = total_heating(quanta)
    band = pred["total_std"]
    return float(np.mean(np.abs(truth - pred["total_mean"]) <= band + 1e-15))


def _metrics(predictor, x, quanta, seed):
    return {
        "log_rmse": log_rmse(predictor, x, quanta, n_samples=48, seed=seed),
        "nll": gaussian_nll(predictor, x, quanta, n_samples=32, seed=seed + 3),
        "coverage": _coverage(predictor, x, quanta, seed + 5),
    }


def _fit(x, quanta, hidden, epochs, seed, lbfgs_maxiter, log_every):
    model, history = train_predictor(
        x,
        quanta,
        kind="stochastic",
        hidden=hidden,
        epochs=epochs,
        seed=seed,
        log_every=log_every,
        noise_samples=8,
        lbfgs_maxiter=lbfgs_maxiter,
        val_fraction=0.2,
    )
    return model, history


def run_study(
    out_dir,
    n=900,
    n_fine=400,
    seeds=(0,),
    fraction=0.55,
    fine_fraction=0.2,
    seed=0,
    epochs=60,
    lbfgs_maxiter=25,
    hidden=32,
    test_fraction=0.2,
    iterations=4,
    n_screen=240,
    n_simulate=8,
):
    """Fill the design range evenly, with a finer cloud around the nominal path.

    The stray field is off. A Latin hypercube covers the wider junction and
    trajectory range, and a second, smaller hypercube fills the neighborhood
    of the nominal path. Active rounds then screen another space-filling pool
    and simulate the low-phonon, high-uncertainty designs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    space = DesignSpace()
    nominal = space.nominal_design()
    if not design_is_admissible(nominal):
        raise RuntimeError("nominal design is outside the 15 um caps")

    print("simulating the nominal design with the stray field off", flush=True)
    baseline = simulate_design(nominal, seeds=seeds, with_ideal=False, noise_std_Vm=0.0)
    if not baseline.ok:
        raise RuntimeError("nominal design failed: {}".format(baseline.message))
    fine = collect_dataset(
        n_fine,
        seeds=seeds,
        fraction=fine_fraction,
        cap_fraction=fine_fraction,
        seed=seed,
        space=space,
        design="latin",
        noise_std_Vm=0.0,
    )
    wide = collect_dataset(
        n,
        seeds=seeds,
        fraction=fraction,
        cap_fraction=1.0,
        seed=seed + 1,
        space=space,
        design="latin",
        noise_std_Vm=0.0,
    )
    initial = _prepend(
        fine,
        space.pack(nominal),
        baseline.quanta,
        baseline.length_m,
        baseline.duration_s,
    )
    initial = initial.extended(wide.x, wide.quanta, wide.length_m, wide.duration_s)
    _check_caps(initial)
    train_idx, test_idx = _split(len(initial), test_fraction, seed)
    if len(train_idx) < 1000:
        raise RuntimeError(
            "training set has {} designs; at least 1000 are required".format(len(train_idx))
        )
    train_data = Dataset(
        initial.x[train_idx],
        initial.quanta[train_idx],
        initial.quanta_ideal[train_idx],
        initial.length_m[train_idx],
        initial.duration_s[train_idx],
        initial.seeds,
        space,
    )
    x_test = initial.x[test_idx]
    q_test = initial.quanta[test_idx]
    print(
        "initial neighborhood: {} train, {} held out, fine {} plus wide {}".format(
            len(train_data), len(test_idx), len(fine) + 1, len(wide)
        ),
        flush=True,
    )

    print("training on the nominal neighborhood", flush=True)
    model, history = _fit(
        train_data.x, train_data.quanta, hidden, epochs, seed, lbfgs_maxiter=0, log_every=40,
    )
    adam = _metrics(model, x_test, q_test, seed)
    print("polishing with L-BFGS", flush=True)
    order = np.random.default_rng(seed + 11).permutation(len(train_data))
    n_val = min(max(int(round(0.2 * len(train_data))), 1), len(train_data) - 2)
    val_i, fit_i = order[:n_val], order[n_val:]
    history["lbfgs"] = lbfgs_polish(
        model.net,
        model.standardize(train_data.x[fit_i]),
        to_log(train_data.quanta[fit_i]),
        maxiter=lbfgs_maxiter,
        noise_samples=8,
        seed=seed + 17,
        x_val=model.standardize(train_data.x[val_i]),
        y_val=to_log(train_data.quanta[val_i]),
    )
    polished = _metrics(model, x_test, q_test, seed)
    rounds = [_round_record(0, model, train_data, x_test, q_test, seed, acquired=None)]

    rng = np.random.default_rng(seed + 23)
    for step in range(1, int(iterations) + 1):
        print("active round {}  space-filling pool over the design range".format(step), flush=True)
        batch, record = acquire(
            model,
            np.vstack([train_data.x, x_test]),
            space,
            rng,
            n_pool=n_screen,
            n_take=n_simulate,
            pool="latin",
            fraction=fraction,
            cap_fraction=1.0,
            seeds=seeds,
            simulate_kwargs={"noise_std_Vm": 0.0},
        )
        if batch is None:
            print("  no admissible candidates", flush=True)
            rounds.append(_round_record(step, model, train_data, x_test, q_test, seed, record))
            continue
        train_data = train_data.extended(*batch)
        _check_caps(train_data)
        print(
            "retraining on {} designs".format(len(train_data)),
            flush=True,
        )
        model, round_history = _fit(
            train_data.x,
            train_data.quanta,
            hidden,
            epochs,
            seed + step,
            lbfgs_maxiter,
            log_every=0,
        )
        history = round_history
        rounds.append(_round_record(step, model, train_data, x_test, q_test, seed, record))
        print(
            "  held-out log10 RMSE {:.4g}  best quanta {:.4g}".format(
                rounds[-1]["log_rmse"], rounds[-1]["best_heating"]
            ),
            flush=True,
        )

    full = train_data.extended(
        x_test,
        q_test,
        np.zeros(len(test_idx)),
        np.zeros(len(test_idx)),
    )
    full.save(out / "dataset.npz")
    model.save(out / "stochastic.npz")
    plot_losses({"stochastic": history}, out / "loss.png")
    plot_prediction(
        model,
        Dataset(
            x_test, q_test, np.full_like(q_test, np.nan),
            np.zeros(len(test_idx)), np.zeros(len(test_idx)), full.seeds, space,
        ),
        out / "prediction.png",
        seed=seed,
    )
    plot_active(rounds, out / "active.png")
    best = int(np.argmin(full.total_mean()))
    best_design = space.unpack(full.x[best])
    if not design_is_admissible(best_design):
        raise RuntimeError("best design left the 15 um caps")
    plot_trajectory(best_design, path_geometry(best_design), out / "best_path.png")

    nominal_heating = float(total_heating(baseline.quanta))
    gap = _gap_from_nominal(space, full, space.pack(nominal), nominal_heating)
    lines = [
        "stray field: off",
        "fine neighborhood: {} designs, fraction {:.2f}, including the nominal path".format(
            len(fine) + 1, fine_fraction
        ),
        "wide range: {} Latin-hypercube designs, fraction {:.2f}, radius and height at the 15 um RF-center cap".format(
            len(wide), fraction
        ),
        "training designs before the search: {}".format(len(train_idx)),
        "training designs after the search: {}".format(len(train_data)),
        "optimization simulations: {}".format(sum(row["n_acquired"] for row in rounds)),
        "held-out designs: {}".format(len(test_idx)),
        "active rounds: {}".format(int(iterations)),
        "designs simulated: {}".format(len(full)),
        "noise seeds: {}".format(len(seeds)),
        "height cap: {:.0f} um from 50 um".format(HEIGHT_CAP_M * 1e6),
        "crossing cap: {:.0f} um".format(CROSS_CAP_M * 1e6),
        "voltage hold: {:.4g} ns".format(baseline.n_hold * SIM_DT * 1e9),
        "nominal total quanta: {:.6g}".format(nominal_heating),
        "Adam held-out log10 RMSE: {:.4g}".format(adam["log_rmse"]),
        "Adam+L-BFGS held-out log10 RMSE: {:.4g}".format(polished["log_rmse"]),
        "Adam held-out Gaussian NLL: {:.4g}".format(adam["nll"]),
        "Adam+L-BFGS held-out Gaussian NLL: {:.4g}".format(polished["nll"]),
        "Adam held-out coverage within 1 std: {:.3f}".format(adam["coverage"]),
        "Adam+L-BFGS held-out coverage within 1 std: {:.3f}".format(polished["coverage"]),
    ]
    for row in rounds:
        lines.append(
            "round {round}  held-out log10 RMSE {log_rmse:.4g}  NLL {nll:.4g}  "
            "coverage {coverage:.3f}  best quanta {best_heating:.6g}  "
            "acquired {n_acquired}".format(**row)
        )
    lines.append("best simulated total quanta: {:.6g}".format(float(full.total_mean()[best])))
    lines.append("best is the nominal design: {}".format("yes" if gap["best_is_nominal"] else "no"))
    lines.append("best unit distance from nominal: {:.3f}".format(gap["best_unit_distance"]))
    lines.append("best path rms deviation: {:.2f} um".format(gap["path_rms_um"]))
    lines.append("designs with lower heating than nominal: {}".format(gap["n_better"]))
    lines.append("median heating within 0.15 of nominal: {:.6g}".format(gap["near_median"]))
    lines.append("median heating farther than 0.15: {:.6g}".format(gap["far_median"]))
    lines.append("lowest five designs, heating and unit distance from nominal:")
    for item in gap["closest_low"]:
        lines.append("  heating {:.6g}  unit distance {:.3f}".format(item[0], item[1]))
    lines.append("best bend radius: {:.2f} um".format(best_design.bend_r * L_SCALE * 1e6))
    lines.append("best crossing distance: {:.2f} um".format(best_design.cross_s * L_SCALE * 1e6))
    lines.append("best crossing height: {:.2f} um".format(best_design.z_cross * L_SCALE * 1e6))
    lines.append("best arm height: {:.2f} to {:.2f} um".format(
        float(best_design.arm_xyz[:, 2].min()) * L_SCALE * 1e6,
        float(best_design.arm_xyz[:, 2].max()) * L_SCALE * 1e6,
    ))
    lines.append("best bend height: {:.2f} to {:.2f} um".format(
        float(min(best_design.z_bend, best_design.interior_xyz[:, 2].min())) * L_SCALE * 1e6,
        float(max(best_design.z_bend, best_design.interior_xyz[:, 2].max())) * L_SCALE * 1e6,
    ))
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return out


def _gap_from_nominal(space, dataset, nominal_vector, nominal_heating):
    """How far the low-heating designs sit from the nominal path."""
    unit = space.to_unit(dataset.x)
    nominal_u = space.to_unit(nominal_vector)
    distance = np.linalg.norm(unit - nominal_u, axis=1)
    totals = dataset.total_mean()
    best = int(np.argmin(totals))
    near = distance <= 0.15
    order = np.argsort(totals)[:5]
    nominal_path = path_geometry(space.unpack(nominal_vector), n=201)
    best_path = path_geometry(space.unpack(dataset.x[best]), n=201)
    rms = float(np.sqrt(np.mean(np.sum((best_path - nominal_path) ** 2, axis=1))))
    return {
        "best_is_nominal": bool(distance[best] < 1e-8),
        "best_unit_distance": float(distance[best]),
        "path_rms_um": rms * L_SCALE * 1e6,
        "n_better": int(np.sum(totals < nominal_heating - 1e-9)),
        "near_median": float(np.median(totals[near])) if np.any(near) else float("nan"),
        "far_median": float(np.median(totals[~near])) if np.any(~near) else float("nan"),
        "closest_low": [(float(totals[i]), float(distance[i])) for i in order],
    }


def _round_record(step, model, train_data, x_test, q_test, seed, acquired):
    metrics = _metrics(model, x_test, q_test, seed)
    archive = train_data.extended(x_test, q_test, np.zeros(len(x_test)), np.zeros(len(x_test)))
    return {
        "round": int(step),
        "log_rmse": metrics["log_rmse"],
        "nll": metrics["nll"],
        "coverage": metrics["coverage"],
        "best_heating": float(np.min(archive.total_mean())),
        "n_acquired": 0 if acquired is None else int(acquired["n_acquired"]),
    }


def _check_caps(dataset):
    for vector in dataset.x:
        design = dataset.space.unpack(vector)
        if not design_is_admissible(design):
            raise RuntimeError("a collected design is outside the 15 um caps")
