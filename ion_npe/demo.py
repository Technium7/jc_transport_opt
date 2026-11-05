"""Short end-to-end run of collection, training, and one search iteration."""

from pathlib import Path

import numpy as np

from .cem import cooptimize
from .constants import SIM_DT
from .dataset import collect_dataset
from .design import DesignSpace
from .plotting import plot_losses, plot_prediction, plot_search, plot_trajectory
from .simulate import simulate_design, total_heating
from .train import log_rmse, train_predictor
from .trajectory import path_geometry


def run_demo(out_dir, seed=0):
    """Collect a small dataset, train both networks, and run one search step.

    The box is kept near the developed junction so the RF curvature stays
    confining. This is a pipeline check, not a converged optimization.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    space = DesignSpace()
    nominal = space.nominal_design()
    print("simulating the nominal design", flush=True)
    baseline = simulate_design(nominal, seeds=(0, 1), with_ideal=False)
    if not baseline.ok:
        raise RuntimeError("nominal design failed: {}".format(baseline.message))
    dataset = collect_dataset(
        6,
        seeds=(0, 1),
        fraction=0.35,
        seed=seed,
        space=space,
    )
    dataset.save(out / "dataset.npz")

    print("training deterministic network", flush=True)
    det, det_hist = train_predictor(
        dataset.x, dataset.quanta, kind="deterministic",
        hidden=32, epochs=200, seed=seed, log_every=50,
    )
    print("training stochastic network", flush=True)
    sto, sto_hist = train_predictor(
        dataset.x, dataset.quanta, kind="stochastic",
        hidden=32, epochs=200, seed=seed, log_every=50, noise_samples=6,
    )
    det.save(out / "deterministic.npz")
    sto.save(out / "stochastic.npz")
    det_rmse = log_rmse(det, dataset.x, dataset.quanta, seed=seed)
    sto_rmse = log_rmse(sto, dataset.x, dataset.quanta, seed=seed)
    plot_losses({"deterministic": det_hist, "stochastic": sto_hist}, out / "loss.png")
    plot_prediction(sto, dataset, out / "prediction.png", seed=seed)

    print("cross-entropy iteration", flush=True)
    _model, dataset, history = cooptimize(
        sto,
        dataset,
        iterations=1,
        n_samples=16,
        n_simulate=2,
        n_elite=4,
        seed=seed,
        finetune_epochs=20,
        noise_samples=6,
    )
    dataset.save(out / "dataset.npz")
    sto.save(out / "stochastic.npz")
    plot_search(history, out / "search.png")

    best = int(np.argmin(dataset.total_mean()))
    best_design = space.unpack(dataset.x[best])
    plot_trajectory(best_design, path_geometry(best_design), out / "best_path.png")
    nominal_heating = float(total_heating(baseline.quanta))
    best_heating = float(dataset.total_mean()[best])
    lines = [
        "nominal total quanta: {:.6g}".format(nominal_heating),
        "best simulated total quanta: {:.6g}".format(best_heating),
        "deterministic log10 RMSE on the initial set: {:.4g}".format(det_rmse),
        "stochastic log10 RMSE on the initial set: {:.4g}".format(sto_rmse),
        "designs simulated: {}".format(len(dataset)),
        "voltage hold: {:.4g} ns".format(baseline.n_hold * SIM_DT * 1e9),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return out
