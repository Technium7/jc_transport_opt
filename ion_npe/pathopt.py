"""Trajectory search over the bend radius and the ``(a, b)`` time profile.

The junction stays at the developed design. Height stays at 50 um. Time-gate
offsets move with the path.
"""

from pathlib import Path

import numpy as np

from .dataset import design_is_admissible
from .design import DesignSpace, N_FREE_GATES, expand_free_gates, free_gate_offsets
from .schedule import schedule_is_admissible
from .simulate import simulation_settings, total_heating
from .train import train_predictor


def _parts(space):
    names = space.names[8:]
    free_columns = np.array([
        name in ("cross_s", "curve_a", "curve_b") for name in names
    ])
    return names, free_columns


def _design_from(space, nominal, trajectory, gates):
    full = np.array(nominal, copy=True)
    full[8:] = trajectory
    design = space.unpack(full)
    design.gate_offsets = expand_free_gates(gates)
    return design


def _ok(design, limit):
    if not design_is_admissible(design):
        return False
    return schedule_is_admissible(
        design.gate_offsets, design.curves, design.n_time_gates, limit=limit
    )


def _draw(rng, n, mean, std, lower, upper, space, nominal, limit):
    drawn_t = []
    drawn_g = []
    tries = 0
    while len(drawn_t) < int(n) and tries < int(n) * 80:
        tries += 1
        trajectory = np.clip(rng.normal(mean[0], std[0]), lower, upper)
        gates = np.clip(rng.normal(mean[1], std[1]), -limit, limit)
        design = _design_from(space, nominal, trajectory, gates)
        if _ok(design, limit):
            drawn_t.append(trajectory)
            drawn_g.append(np.asarray(gates, dtype=float).copy())
    if len(drawn_t) < int(n):
        raise RuntimeError("could not draw enough admissible trajectories")
    return np.vstack(drawn_t), np.vstack(drawn_g)


def _reference_rows(nominal_traj, gates):
    rows_t = [nominal_traj.copy()]
    rows_g = [np.zeros_like(gates)]
    if np.any(gates):
        rows_t.append(nominal_traj.copy())
        rows_g.append(np.asarray(gates, dtype=float).copy())
    return rows_t, rows_g


def optimize_trajectory(
    out,
    n=120,
    iterations=6,
    n_screen=48,
    n_simulate=6,
    epochs=40,
    lbfgs_maxiter=20,
    seed=0,
    offset_limit=0.10,
    simulate_kwargs=None,
):
    """Search the bend radius and ``(a, b)``. Height stays at 50 um."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    simulate_kwargs = dict(simulate_kwargs or {})
    simulate_kwargs = simulation_settings(simulate_kwargs)
    space = DesignSpace()
    nominal = space.nominal()
    names, free_columns = _parts(space)
    lower = space.lower[8:]
    upper = space.upper[8:]
    span = upper - lower
    nominal_traj = nominal[8:]
    saved = Path("outputs/schedule/schedule.npz")
    if saved.exists():
        loaded = np.asarray(np.load(saved)["best_offsets"], dtype=float).reshape(-1)
        if loaded.size in (N_FREE_GATES, 7):
            best_gates = free_gate_offsets(loaded)
        else:
            best_gates = np.zeros(N_FREE_GATES)
    else:
        best_gates = np.zeros(N_FREE_GATES)
    rng = np.random.default_rng(seed)

    probe_t, probe_g = _reference_rows(nominal_traj, best_gates)
    keep_t = []
    keep_g = []
    for trajectory, gates in zip(probe_t, probe_g):
        design = _design_from(space, nominal, trajectory, gates)
        if _ok(design, offset_limit):
            keep_t.append(np.clip(trajectory, lower, upper))
            keep_g.append(np.asarray(gates, dtype=float).copy())
    unit_std = np.where(free_columns, 0.28, 0.06)
    cloud_t, cloud_g = _draw(
        rng,
        max(int(n) - len(keep_t), 0),
        (nominal_traj, np.zeros_like(best_gates)),
        (unit_std * span, np.full(best_gates.shape, 0.025)),
        lower,
        upper,
        space,
        nominal,
        offset_limit,
    )
    trajectories = np.vstack(keep_t + [cloud_t])
    gates = np.vstack(keep_g + [cloud_g])
    print("collecting {} trajectories, junction fixed".format(len(trajectories)), flush=True)

    from .farm import get_farm

    joint_rows = []
    for index in range(len(trajectories)):
        full = np.array(nominal, copy=True)
        full[8:] = trajectories[index]
        joint_rows.append(np.concatenate([full, gates[index]]))
    packed = get_farm().simulate("joint", np.vstack(joint_rows), simulate_kwargs)
    quanta = []
    totals = []
    for index, result in enumerate(packed):
        if not result["ok"]:
            raise RuntimeError(result["message"])
        heating = float(total_heating(result["quanta"]))
        quanta.append(result["quanta"])
        totals.append(heating)
        print("  {:4d}  heating {:.4g}".format(index + 1, heating), flush=True)
    quanta = np.stack(quanta)
    totals = np.asarray(totals, dtype=float)
    features = np.hstack([trajectories, gates])

    def fit(x, y, fit_seed):
        predictor, _history = train_predictor(
            x, y, kind="stochastic", hidden=32, epochs=epochs, seed=fit_seed,
            val_fraction=0.2, lbfgs_maxiter=lbfgs_maxiter, log_every=0,
        )
        return predictor

    predictor = fit(features, quanta, seed)
    for iteration in range(int(iterations)):
        order = np.argsort(totals)[: max(n_simulate, 4)]
        mean_t = trajectories[order].mean(axis=0)
        std_t = np.maximum(trajectories[order].std(axis=0), 0.15 * unit_std * span)
        mean_g = gates[order].mean(axis=0)
        std_g = np.maximum(gates[order].std(axis=0), 0.01)
        pool_t, pool_g = _draw(
            rng, n_screen, (mean_t, mean_g), (std_t, std_g),
            lower, upper, space, nominal, offset_limit,
        )
        pool = np.hstack([pool_t, pool_g])
        pred = predictor.predict_heating(pool, n_samples=32, rng=rng)
        chosen = np.argsort(pred["total_mean"])[:n_simulate]
        print(
            "trajectory round {}  simulating {}".format(iteration + 1, len(chosen)),
            flush=True,
        )
        from .farm import get_farm

        joint_rows = []
        for index in chosen:
            full = np.array(nominal, copy=True)
            full[8:] = pool_t[index]
            joint_rows.append(np.concatenate([full, pool_g[index]]))
        packed = get_farm().simulate("joint", np.vstack(joint_rows), simulate_kwargs)
        new_q = []
        new_total = []
        kept_t = []
        kept_g = []
        for index, result in zip(chosen, packed):
            if not result["ok"]:
                print("  skip: {}".format(result["message"]), flush=True)
                continue
            heating = float(total_heating(result["quanta"]))
            print("  simulated heating {:.4g}".format(heating), flush=True)
            new_q.append(result["quanta"])
            new_total.append(heating)
            kept_t.append(pool_t[index])
            kept_g.append(pool_g[index])
        if not new_q:
            continue
        trajectories = np.vstack([trajectories, np.vstack(kept_t)])
        gates = np.vstack([gates, np.vstack(kept_g)])
        quanta = np.concatenate([quanta, np.stack(new_q)], axis=0)
        totals = np.concatenate([totals, np.asarray(new_total, dtype=float)])
        features = np.hstack([trajectories, gates])
        print("  best so far {:.4g}".format(float(np.min(totals))), flush=True)
        predictor = fit(features, quanta, seed + iteration + 1)

    best_i = int(np.argmin(totals))
    design = _design_from(space, nominal, trajectories[best_i], gates[best_i])
    from .plotting import plot_best_design, plot_trajectory

    plot_trajectory(design, None, out / "best_path.png")
    plot_best_design(
        design, out / "best_design.png",
        reference=space.nominal_design(), total_quanta=float(totals[best_i]),
    )
    np.savez(
        out / "trajectory.npz",
        trajectories=trajectories, gates=gates, quanta=quanta, totals=totals,
        names=np.array(names),
    )
    lines = [
        "junction fixed at the developed design",
        "start and end height: 50 um",
        "trajectories simulated: {}".format(len(totals)),
        "nominal total quanta: {:.6g}".format(float(totals[0])),
        "best total quanta: {:.6g}".format(float(totals[best_i])),
        "best quanta: {}".format(
            " ".join("{:.6g}".format(float(v)) for v in quanta[best_i].reshape(-1))
        ),
        "best bend entrance: {:.2f} um".format(design.z_bend * 50.0),
        "best crossing height: {:.2f} um".format(design.z_cross * 50.0),
        "best bend radius s: {:.2f} um".format(design.cross_s * 50.0),
        "best bend control height: {:.2f} to {:.2f} um".format(
            design.interior_z.min() * 50.0, design.interior_z.max() * 50.0
        ),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return {"best": float(totals[best_i]), "nominal": float(totals[0])}
