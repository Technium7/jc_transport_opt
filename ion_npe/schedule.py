"""Waveform-gate search, matching the straight-line transport optimizer.

That search cut the phonon number by about 10x by moving five intermediate
gates of an otherwise fixed transport and keeping the cross-entropy cloud on
the designs that had just been simulated. The junction and the spatial bend
stay at the nominal design. Only the interior samples of the time law move.
"""

from pathlib import Path

import numpy as np

from .design import DesignSpace, symmetric_gate_offsets
from .simulate import simulate_design, simulation_settings, stray_field_line, total_heating
from .train import train_predictor
from .trajectory import s_of_tau, time_gate_samples


def schedule_is_admissible(offsets, curves, n_gates, limit=0.10, gap=0.015):
    """Interior gates stay near the nominal waveform and the ion never reverses."""
    offsets = np.asarray(offsets, dtype=float).reshape(-1)
    if offsets.size != int(n_gates) - 1:
        return False
    if np.any(np.abs(offsets) > float(limit) + 1e-12):
        return False
    _knots, samples = time_gate_samples(curves, n_gates, offsets)
    if np.any(np.diff(samples) < float(gap)):
        return False
    tau = np.linspace(0.0, 1.0, 241)
    distance = s_of_tau(tau, curves, n_gates, offsets)
    if distance[0] < -1e-6 or distance[-1] > 1.0 + 1e-6:
        return False
    return bool(np.all(np.diff(distance) >= -1e-8))


def _draw_offsets(rng, n, mean, std, curves, n_gates, limit):
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.broadcast_to(np.asarray(std, dtype=float), mean.shape).copy()
    if int(n) <= 0:
        return np.zeros((0, mean.size))
    drawn = []
    tries = 0
    cap = max(int(n) * 80, 80)
    while len(drawn) < int(n) and tries < cap:
        tries += 1
        offsets = np.clip(rng.normal(mean, std), -limit, limit)
        offsets = symmetric_gate_offsets(offsets)
        if schedule_is_admissible(offsets, curves, n_gates, limit=limit):
            drawn.append(offsets)
    if len(drawn) < int(n):
        raise RuntimeError("could not draw enough monotone transport waveforms")
    return np.vstack(drawn)


def _probe_offsets(step, n_gates):
    """Paired gate moves, so every probe waveform is symmetric about mid-path."""
    n_interior = int(n_gates) - 1
    rows = [np.zeros(n_interior)]
    for index in range(n_interior // 2):
        for sign in (-1.0, 1.0):
            for scale in (0.5, 1.0):
                row = np.zeros(n_interior)
                row[index] = sign * scale * float(step)
                row[-1 - index] = -row[index]
                rows.append(row)
    return np.vstack(rows)


def _with_offsets(design, offsets):
    design.gate_offsets = symmetric_gate_offsets(offsets)
    return design


def _simulate_rows(base, offsets, simulate_kwargs):
    quanta = []
    totals = []
    for index, row in enumerate(offsets):
        design = _with_offsets(base, row)
        result = simulate_design(design, **simulate_kwargs)
        if not result.ok:
            raise RuntimeError(result.message)
        heating = float(total_heating(result.quanta))
        quanta.append(result.quanta)
        totals.append(heating)
        print(
            "  {:4d}  heating {:.4g}  quanta {}".format(
                index + 1,
                heating,
                " ".join("{:.4g}".format(float(v)) for v in result.quanta.reshape(-1)),
            ),
            flush=True,
        )
    return np.stack(quanta), np.asarray(totals, dtype=float)


def _plot_time_law(curves, n_gates, best_offsets, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = np.linspace(0.0, 1.0, 400)
    nominal = s_of_tau(tau, curves, n_gates)
    best = s_of_tau(tau, curves, n_gates, best_offsets)
    knots, nominal_gates = time_gate_samples(curves, n_gates)
    _knots, best_gates = time_gate_samples(curves, n_gates, best_offsets)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(tau, nominal, color="#4C78A8", label="nominal waveform")
    ax.plot(tau, best, color="#E45756", label="optimized gates")
    ax.scatter(knots, nominal_gates, s=28, color="#4C78A8", zorder=3)
    ax.scatter(knots[1:-1], best_gates[1:-1], s=36, facecolors="none", edgecolors="#E45756", zorder=4)
    ax.set_xlabel("normalized time")
    ax.set_ylabel("arc-length fraction")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def optimize_schedule(
    out,
    n=180,
    iterations=8,
    n_screen=64,
    n_simulate=8,
    epochs=60,
    lbfgs_maxiter=25,
    seed=0,
    offset_limit=0.10,
    offset_std=0.025,
    simulate_kwargs=None,
):
    """Train on waveforms near the nominal path, then cross-entropy on the gates.

    The junction, the bend, and ``(a, b)`` stay at the developed design. The
    five interior time gates move, with the same role as the intermediate
    position gates in the straight-line search.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    simulate_kwargs = simulation_settings(simulate_kwargs)
    space = DesignSpace()
    base = space.nominal_design()
    curves = base.curves
    n_gates = int(base.n_time_gates)
    rng = np.random.default_rng(seed)

    print("nominal waveform, {}".format(stray_field_line(simulate_kwargs)), flush=True)
    nominal_q, nominal_total = _simulate_rows(base, np.zeros((1, n_gates - 1)), simulate_kwargs)
    baseline = float(nominal_total[0])

    probes = _probe_offsets(0.04, n_gates)
    keep = [
        row for row in probes
        if schedule_is_admissible(row, curves, n_gates, limit=offset_limit)
    ]
    cloud = _draw_offsets(
        rng,
        max(int(n) - len(keep), 0),
        np.zeros(n_gates - 1),
        offset_std,
        curves,
        n_gates,
        offset_limit,
    )
    offsets = np.vstack(keep + [cloud] if len(cloud) else keep)
    # The zero waveform is already row 0 of the probes.
    print(
        "collecting {} waveforms, gate std {:.3f} of the path".format(len(offsets), offset_std),
        flush=True,
    )
    if len(offsets) > 1:
        rest_q, rest_total = _simulate_rows(base, offsets[1:], simulate_kwargs)
        quanta = np.concatenate([nominal_q, rest_q], axis=0)
        totals = np.concatenate([nominal_total, rest_total])
    else:
        quanta, totals = nominal_q, nominal_total
    n_before = len(offsets)

    def fit(x, y, fit_seed):
        predictor, _history = train_predictor(
            x,
            y,
            kind="stochastic",
            hidden=32,
            epochs=epochs,
            seed=fit_seed,
            val_fraction=0.2,
            lbfgs_maxiter=lbfgs_maxiter,
            log_every=0,
        )
        return predictor

    print("training on the waveform gates", flush=True)
    predictor = fit(offsets, quanta, seed)
    rounds = []
    best_total = float(np.min(totals))
    for iteration in range(int(iterations)):
        order = np.argsort(totals)
        elite = offsets[order[: max(n_simulate, 4)]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.008)
        pool = _draw_offsets(rng, n_screen, mean, std, curves, n_gates, offset_limit)
        pred = predictor.predict_heating(pool, n_samples=32, rng=rng)
        chosen = pool[np.argsort(pred["total_mean"])[:n_simulate]]
        print(
            "schedule round {}  simulating {} gates near the current best".format(
                iteration + 1, len(chosen)
            ),
            flush=True,
        )
        new_q, new_total = _simulate_rows(base, chosen, simulate_kwargs)
        offsets = np.vstack([offsets, chosen])
        quanta = np.concatenate([quanta, new_q], axis=0)
        totals = np.concatenate([totals, new_total])
        best_total = float(np.min(totals))
        print("  best so far {:.4g}  nominal {:.4g}".format(best_total, baseline), flush=True)
        predictor = fit(offsets, quanta, seed + iteration + 1)
        rounds.append((iteration + 1, best_total, int(len(offsets))))

    best_i = int(np.argmin(totals))
    best_offsets = offsets[best_i]
    best_quanta = quanta[best_i, 0]
    design = _with_offsets(base, best_offsets)
    length_m = float(simulate_design(design, **simulate_kwargs).length_m)
    _plot_time_law(curves, n_gates, best_offsets, out / "time_law.png")
    from .plotting import plot_best_design

    plot_best_design(
        design,
        out / "best_design.png",
        reference=space.nominal_design(),
        total_quanta=best_total,
    )
    np.savez(
        out / "schedule.npz",
        offsets=offsets,
        quanta=quanta,
        totals=totals,
        best_offsets=best_offsets,
    )
    factor = baseline / best_total if best_total > 0.0 else np.inf
    lines = [
        "junction and bend fixed at the nominal design",
        stray_field_line(simulate_kwargs),
        "interior time gates: {}".format(n_gates - 1),
        "training waveforms before the search: {}".format(n_before),
        "waveforms simulated: {}".format(len(offsets)),
        "nominal quanta: {}".format(" ".join("{:.6g}".format(float(v)) for v in nominal_q.reshape(-1))),
        "nominal total quanta: {:.6g}".format(baseline),
        "best quanta: {}".format(" ".join("{:.6g}".format(float(v)) for v in best_quanta.reshape(-1))),
        "best total quanta: {:.6g}".format(best_total),
        "reduction factor: {:.4g}".format(factor),
        "best gate shifts (um along the path): {}".format(
            " ".join("{:.3g}".format(float(v) * length_m * 1e6) for v in best_offsets)
        ),
    ]
    for iteration, heating, count in rounds:
        lines.append(
            "round {}  best quanta {:.6g}  waveforms {}".format(iteration, heating, count)
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return {
        "baseline": baseline,
        "best": best_total,
        "factor": factor,
        "offsets": best_offsets,
    }
