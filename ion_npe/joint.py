"""One search over the junction, the trajectory, and the waveform.

The space-filling sample trains the network. Cross-entropy rounds then
simulate the designs that network ranks as coldest. Bend-control heights stay
within 15 um of 50 um. The free trajectory coordinates are the bend radius,
those heights, and ``(a, b)``. The waveform coordinates are the four outbound
gates. The center gate stays at 0, and the return half is their mirror.
"""

from pathlib import Path

import numpy as np

from .constants import L_SCALE
from .dataset import design_is_admissible
from .design import (
    DesignSpace,
    N_FREE_GATES,
    expand_free_gates,
    free_gate_offsets,
)
from .schedule import schedule_is_admissible
from .simulate import (
    force_line,
    heating_line,
    simulate_design,
    simulation_settings,
    stray_field_line,
    total_heating,
)
from .active import active_ranking
from .train import finetune, train_predictor

# The spline has nine interior gates. The search stores the four outbound
# gates. The center gate is fixed, and the return half mirrors them.
N_GATES = 2 * N_FREE_GATES + 1
GATE_LIMIT = 0.10


def _bounds(space):
    lower = np.concatenate([space.lower, np.full(N_FREE_GATES, -GATE_LIMIT)])
    upper = np.concatenate([space.upper, np.full(N_FREE_GATES, GATE_LIMIT)])
    return lower, upper


def _std(space, lower, upper, gate_std):
    span = upper - lower
    names = list(space.names) + ["g{}".format(i) for i in range(N_FREE_GATES)]
    std = np.empty(len(names))
    for index, name in enumerate(names):
        if name.startswith("g"):
            std[index] = gate_std
        elif name == "cross_s" or name.endswith("_z"):
            std[index] = 0.22 * span[index]
        elif name in ("curve_a", "curve_b"):
            std[index] = 0.12 * span[index]
        else:
            std[index] = 0.08 * span[index]
    return std


def project_joint_row(row):
    """Joint search vector: the design, then the four outbound gates.

    A saved row that still carries all nine spline gates is reduced. The
    center gate is dropped and the return half is not stored again.
    """
    values = np.asarray(row, dtype=float).reshape(-1)
    dim = DesignSpace().dim
    if values.size == dim + N_GATES:
        gates = free_gate_offsets(values[dim:])
    elif values.size == dim + N_FREE_GATES:
        gates = free_gate_offsets(values[dim:])
    else:
        raise ValueError(
            "joint vector must have length {} or {}, got {}".format(
                dim + N_FREE_GATES, dim + N_GATES, values.size
            )
        )
    return np.concatenate([values[:dim], gates])


def _unpack(space, row):
    prepared = project_joint_row(row)
    design = space.unpack(prepared[:-N_FREE_GATES])
    design.gate_offsets = expand_free_gates(prepared[-N_FREE_GATES:])
    return design


def _admissible(design):
    if not design_is_admissible(design):
        return False
    return schedule_is_admissible(
        design.gate_offsets, design.curves, design.n_time_gates, limit=GATE_LIMIT
    )


def _saved_gates():
    path = Path("outputs/schedule/schedule.npz")
    if not path.exists():
        return np.zeros(N_FREE_GATES)
    gates = np.asarray(np.load(path)["best_offsets"], dtype=float).reshape(-1)
    if gates.size not in (N_FREE_GATES, N_GATES):
        return np.zeros(N_FREE_GATES)
    return free_gate_offsets(gates)


def _simulate_row(space, row, simulate_kwargs):
    design = _unpack(space, row)
    if not _admissible(design):
        return None
    result = simulate_design(design, **simulate_kwargs)
    if not result.ok:
        print("  skip: {}".format(result.message), flush=True)
        return None
    return result


def _draw(rng, n, mean, std, lower, upper, space):
    rows = []
    tries = 0
    cap = max(int(n) * 80, 80)
    while len(rows) < int(n) and tries < cap:
        tries += 1
        row = np.clip(rng.normal(mean, std), lower, upper)
        prepared = project_joint_row(row)
        if _admissible(_unpack(space, prepared)):
            rows.append(prepared)
    if len(rows) < int(n):
        raise RuntimeError("could not draw {} admissible joint designs".format(n))
    return np.vstack(rows)


def _extend_joint(rows, quanta, totals, prepared, packed, limit, label, log_every):
    """Append successful integrations, in order, until ``limit`` rows are stored."""
    for row, result in zip(prepared, packed):
        if len(rows) >= int(limit):
            break
        if not result["ok"]:
            if result["message"]:
                print("  skip: {}".format(result["message"]), flush=True)
            continue
        heating = float(total_heating(result["quanta"]))
        rows.append(np.asarray(row, dtype=float))
        quanta.append(result["quanta"])
        totals.append(heating)
        if log_every <= 1 or len(rows) == 1 or len(rows) % int(log_every) == 0 or len(rows) == int(limit):
            print(
                "  {} {:4d}/{}  heating {:.4g}".format(label, len(rows), int(limit), heating),
                flush=True,
            )
    return rows, quanta, totals


def _collect_latin(rng, n, lower, upper, space, simulate_kwargs):
    """Space-filling joint designs. Each draw moves the junction, the path, and the gates."""
    from scipy.stats.qmc import LatinHypercube

    from .farm import get_farm, width_for

    farm = get_farm()
    rows = []
    quanta = []
    totals = []
    queue = []
    while len(rows) < int(n):
        if not queue:
            batch_n = max((int(n) - len(rows)) * 2, 64)
            unit = LatinHypercube(d=len(lower), seed=rng).random(batch_n)
            batch = lower + unit * (upper - lower)
            for row in batch:
                prepared = project_joint_row(row)
                if _admissible(_unpack(space, prepared)):
                    queue.append(prepared)
            if not queue:
                continue
        need = int(n) - len(rows)
        take = min(len(queue), width_for(need, farm.size))
        chunk = queue[:take]
        del queue[:take]
        packed = farm.simulate("joint", np.vstack(chunk), simulate_kwargs)
        rows, quanta, totals = _extend_joint(
            rows, quanta, totals, chunk, packed, n, "varied", log_every=50,
        )
    return np.vstack(rows), np.stack(quanta), np.asarray(totals, dtype=float)


def _collect(rng, n, mean, std, lower, upper, space, simulate_kwargs, label):
    from .farm import get_farm, width_for

    farm = get_farm()
    rows = []
    quanta = []
    totals = []
    pending = []
    tries = 0
    cap = max(int(n) * 40, 40)

    def _flush():
        nonlocal rows, quanta, totals
        if not pending:
            return
        packed = farm.simulate("joint", np.vstack(pending), simulate_kwargs)
        rows, quanta, totals = _extend_joint(
            rows, quanta, totals, pending, packed, n, label, log_every=1,
        )
        pending.clear()

    while len(rows) < int(n) and tries < cap:
        if pending and len(pending) >= width_for(int(n) - len(rows), farm.size):
            _flush()
            continue
        tries += 1
        row = np.clip(rng.normal(mean, std), lower, upper)
        prepared = project_joint_row(row)
        if not _admissible(_unpack(space, prepared)):
            continue
        pending.append(prepared)
    _flush()
    if len(rows) < int(n):
        raise RuntimeError("could not simulate {} joint designs".format(n))
    return np.vstack(rows), np.stack(quanta), np.asarray(totals, dtype=float)


def _load_cache(path):
    data = np.load(path)
    n_train = int(np.asarray(data["n_train"])) if "n_train" in data.files else None
    if "origin" in data.files:
        origin = np.asarray(data["origin"]).astype(str)
    else:
        origin = None
    return (
        np.asarray(data["rows"], dtype=float),
        np.asarray(data["quanta"], dtype=float),
        np.asarray(data["totals"], dtype=float),
        n_train,
        origin,
    )


def _save_cache(path, rows, quanta, totals, n_train, origin):
    np.savez(
        path,
        rows=rows,
        quanta=quanta,
        totals=totals,
        n_train=np.int64(n_train),
        origin=np.asarray(origin, dtype="U16"),
    )


def _location(predictor, pool, rng, n_samples=32):
    """Predicted total quanta and the std of its ``log10``, one row per design."""
    logs = predictor.predict_log_samples(pool, n_samples=n_samples, rng=rng)
    total = np.maximum(np.sum(10.0 ** logs, axis=-1), 1e-30)
    log_total = np.log10(total)
    return 10.0 ** log_total.mean(axis=0), np.maximum(log_total.std(axis=0), 1e-6)


def _novel(row, existing, lower, upper, tol=0.03):
    if len(existing) == 0:
        return True
    span = np.maximum(upper - lower, 1e-8)
    unit = (row - lower) / span
    have = (existing - lower) / span
    return float(np.min(np.linalg.norm(have - unit, axis=1))) >= tol


def _pick_lowest(scores, n):
    """Indices of the ``n`` smallest scores, best first."""
    order = np.argsort(np.asarray(scores, dtype=float))
    return [int(i) for i in order[: int(n)]]


def _latin_pool(rng, n, lower, upper, space):
    from scipy.stats.qmc import LatinHypercube

    rows = []
    while len(rows) < int(n):
        batch_n = max((int(n) - len(rows)) * 2, 64)
        unit = LatinHypercube(d=len(lower), seed=rng).random(batch_n)
        batch = lower + unit * (upper - lower)
        for row in batch:
            if len(rows) >= int(n):
                break
            prepared = project_joint_row(row)
            if _admissible(_unpack(space, prepared)):
                rows.append(prepared)
    return np.vstack(rows)


def _simulate_ordered(space, pool, order, predicted, existing, lower, upper, n_take, simulate_kwargs, label):
    """Integrate novel rows in ``order`` until ``n_take`` succeed."""
    from .farm import get_farm, width_for

    farm = get_farm()
    kept = []
    quanta = []
    totals = []
    pending_index = []
    pending_row = []

    def _flush():
        nonlocal existing
        if not pending_row:
            return
        packed = farm.simulate("joint", np.vstack(pending_row), simulate_kwargs)
        for index, row, result in zip(pending_index, pending_row, packed):
            if len(kept) >= int(n_take):
                break
            if not _novel(row, existing, lower, upper):
                continue
            if not result["ok"]:
                if result["message"]:
                    print("  skip: {}".format(result["message"]), flush=True)
                continue
            heating = float(total_heating(result["quanta"]))
            print(
                "  {} heating {:.4g}  predicted {:.4g}".format(
                    label, heating, float(predicted[index])
                ),
                flush=True,
            )
            kept.append(row)
            quanta.append(np.asarray(result["quanta"], dtype=float))
            totals.append(heating)
            existing = np.vstack([existing, np.asarray(row, dtype=float).reshape(1, -1)])
        pending_index.clear()
        pending_row.clear()

    for index in order:
        if len(kept) >= int(n_take):
            break
        row = project_joint_row(pool[index])
        if not _novel(row, existing, lower, upper):
            continue
        if not _admissible(_unpack(space, row)):
            continue
        pending_index.append(int(index))
        pending_row.append(row)
        if len(pending_row) >= width_for(int(n_take) - len(kept), farm.size):
            _flush()
    _flush()
    return kept, quanta, totals


def _append(rows, quanta, totals, origin, kept, new_q, new_total, tag):
    if not kept:
        return rows, quanta, totals, origin
    stacked = [np.asarray(q, dtype=float).reshape(1, -1, 3) for q in new_q]
    return (
        np.vstack([rows, np.vstack(kept)]),
        np.concatenate([quanta, np.concatenate(stacked, axis=0)], axis=0),
        np.concatenate([totals, np.asarray(new_total, dtype=float)]),
        np.concatenate([origin, np.full(len(kept), tag, dtype="U16")]),
    )


def _shifts(row, nominal, space):
    unit = space.to_unit(row[:-N_FREE_GATES])
    base = space.to_unit(nominal[:-N_FREE_GATES])
    junction = float(np.linalg.norm(unit[:8] - base[:8]))
    path = float(np.linalg.norm(unit[8:] - base[8:]))
    gates = float(np.linalg.norm(row[-N_FREE_GATES:] - nominal[-N_FREE_GATES:]))
    return junction, path, gates


def _cache_is_ready(path, n):
    if not path.exists():
        return False
    rows, _quanta, _totals, n_train, _origin = _load_cache(path)
    have = len(rows) if n_train is None else max(len(rows), n_train)
    return have >= int(n) and rows.shape[1] == len(_bounds(DesignSpace())[0])


def optimize_joint(
    out,
    n=5000,
    iterations=6,
    active_rounds=4,
    n_screen=256,
    n_simulate=8,
    n_elite=16,
    epochs=40,
    lbfgs_maxiter=20,
    seed=0,
    simulate_kwargs=None,
):
    """Train on the joint sample, then add CEM and active-learning simulations.

    The Latin hypercube is the training set. It is resimulated only when a
    saved sample of at least ``n`` designs is not already on disk. Each CEM
    round draws an elite cloud and a neighborhood of the best design, both of
    which move the junction, the path, and the gates, and integrates the
    designs the network ranks lowest in each cloud. Each active round screens
    a fresh space-filling pool and integrates the low-heating, high-uncertainty
    designs.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    simulate_kwargs = simulation_settings(simulate_kwargs)
    space = DesignSpace()
    lower, upper = _bounds(space)
    nominal = np.concatenate([space.nominal(), np.zeros(N_FREE_GATES)])
    cache = out / "joint.npz"
    rng = np.random.default_rng(seed)
    local_std = _std(space, lower, upper, 0.03)

    if _cache_is_ready(cache, n):
        rows, quanta, totals, n_train, origin = _load_cache(cache)
        if quanta.ndim != 3 or quanta.shape[1] != len(simulate_kwargs["seeds"]):
            raise RuntimeError(
                "saved joint designs have {} stray-field samples; this run uses {}".format(
                    quanta.shape[1] if quanta.ndim == 3 else 1, len(simulate_kwargs["seeds"])
                )
            )
        if n_train is None:
            n_train = len(rows)
        if origin is None or len(origin) != len(rows):
            origin = np.array(["train"] * len(rows), dtype="U16")
        print(
            "loaded {} joint designs, {} of them the training set".format(len(rows), n_train),
            flush=True,
        )
    else:
        waveform = nominal.copy()
        waveform[-N_FREE_GATES:] = _saved_gates()
        print("nominal junction, trajectory, and waveform", flush=True)
        nominal_result = _simulate_row(space, nominal, simulate_kwargs)
        if nominal_result is None:
            raise RuntimeError("nominal joint design is not admissible")
        pieces = [nominal.reshape(1, -1)]
        quanta_parts = [nominal_result.quanta.reshape(1, *nominal_result.quanta.shape)]
        total_parts = [float(total_heating(nominal_result.quanta))]
        print("  heating {:.4g}".format(total_parts[-1]), flush=True)
        if np.any(waveform[-N_FREE_GATES:]):
            print("same junction and path, reshaped waveform", flush=True)
            wave_result = _simulate_row(space, waveform, simulate_kwargs)
            if wave_result is not None:
                pieces.append(waveform.reshape(1, -1))
                quanta_parts.append(wave_result.quanta.reshape(1, *wave_result.quanta.shape))
                total_parts.append(float(total_heating(wave_result.quanta)))
                print("  heating {:.4g}".format(total_parts[-1]), flush=True)
        print(
            "latin sample of {} designs over the junction, the trajectory, and the waveform".format(int(n)),
            flush=True,
        )
        drawn, drawn_q, drawn_total = _collect_latin(
            rng, int(n), lower, upper, space, simulate_kwargs
        )
        pieces.append(drawn)
        quanta_parts.append(drawn_q)
        total_parts.append(drawn_total)
        rows = np.vstack(pieces)
        quanta = np.concatenate(quanta_parts, axis=0)
        totals = np.concatenate([np.atleast_1d(item) for item in total_parts])
        n_train = len(rows)
        origin = np.array(["train"] * n_train, dtype="U16")
        _save_cache(cache, rows, quanta, totals, n_train, origin)
        print("saved {} joint designs".format(len(rows)), flush=True)

    kind = "stochastic" if quanta.shape[1] >= 2 else "deterministic"
    print(
        "training a {} network on {} joint designs".format(kind, len(rows)),
        flush=True,
    )
    predictor, _history = train_predictor(
        rows, quanta, kind=kind, hidden=48, epochs=epochs, seed=seed,
        val_fraction=0.2, lbfgs_maxiter=lbfgs_maxiter, log_every=20, batch_size=64,
    )
    n_cem_each = max(int(n_simulate) // 2, 1)
    for iteration in range(int(iterations)):
        elite = np.argsort(totals)[: max(int(n_elite), 2)]
        mean = rows[elite].mean(axis=0)
        std = np.maximum(rows[elite].std(axis=0), 0.5 * local_std)
        best = rows[int(np.argmin(totals))]
        print(
            "CEM round {}  elite cloud and a neighborhood of the best design".format(iteration + 1),
            flush=True,
        )
        elite_pool = _draw(rng, max(int(n_screen) // 2, n_cem_each), mean, std, lower, upper, space)
        best_pool = _draw(rng, max(int(n_screen) // 2, n_cem_each), best, local_std, lower, upper, space)
        for pool, tag, take in (
            (elite_pool, "cem-elite", n_cem_each),
            (best_pool, "cem-best", n_cem_each),
        ):
            predicted, _log_std = _location(predictor, pool, rng)
            kept, new_q, new_total = _simulate_ordered(
                space, pool, _pick_lowest(predicted, len(predicted)), predicted,
                rows, lower, upper, take, simulate_kwargs, tag,
            )
            rows, quanta, totals, origin = _append(
                rows, quanta, totals, origin, kept, new_q, new_total, tag,
            )
        print("  best so far {:.4g} over {} simulations".format(float(np.min(totals)), len(totals)), flush=True)
        _save_cache(cache, rows, quanta, totals, n_train, origin)
        finetune(
            predictor, rows, quanta, epochs=max(epochs // 2, 10), seed=seed + iteration + 1,
            log_every=0, lbfgs_maxiter=lbfgs_maxiter, batch_size=64,
        )

    for step in range(int(active_rounds)):
        print(
            "active round {}  space-filling pool over the junction, the trajectory, and the waveform".format(step + 1),
            flush=True,
        )
        pool = _latin_pool(rng, int(n_screen), lower, upper, space)
        predicted, log_std = _location(predictor, pool, rng)
        order = active_ranking(predicted, 10.0 ** log_std, phonon_quantile=0.5)
        kept, new_q, new_total = _simulate_ordered(
            space, pool, order, predicted, rows, lower, upper,
            int(n_simulate), simulate_kwargs, "active",
        )
        rows, quanta, totals, origin = _append(
            rows, quanta, totals, origin, kept, new_q, new_total, "active",
        )
        print("  best so far {:.4g} over {} simulations".format(float(np.min(totals)), len(totals)), flush=True)
        _save_cache(cache, rows, quanta, totals, n_train, origin)
        if kept:
            finetune(
                predictor, rows, quanta, epochs=max(epochs // 2, 10), seed=seed + 100 + step,
                log_every=0, lbfgs_maxiter=lbfgs_maxiter, batch_size=64,
            )
    predictor.save(out / "stochastic.npz")

    train_rows = np.flatnonzero(origin == "train")
    n_ref = 2 if len(train_rows) > 1 and np.any(rows[train_rows[1], -N_FREE_GATES:]) else 1
    latin = train_rows[n_ref:]
    cem = np.flatnonzero((origin == "cem-elite") | (origin == "cem-best"))
    active = np.flatnonzero(origin == "active")
    added = np.concatenate([cem, active]) if len(cem) or len(active) else np.array([], dtype=int)
    plot_i = int(added[np.argmin(totals[added])]) if len(added) else int(latin[np.argmin(totals[latin])])
    design = _unpack(space, rows[plot_i])
    junction_shift, path_shift, gate_shift = _shifts(rows[plot_i], nominal, space)
    from .plotting import plot_best_design, plot_trajectory

    plot_trajectory(design, None, out / "best_path.png")
    plot_best_design(
        design, out / "best_design.png",
        reference=space.nominal_design(), total_quanta=float(totals[plot_i]),
    )
    _save_cache(cache, rows, quanta, totals, n_train, origin)
    unit = space.to_unit(rows[latin, :-N_FREE_GATES])
    gate_unit = (rows[latin, -N_FREE_GATES:] - lower[-N_FREE_GATES:]) / (
        upper[-N_FREE_GATES:] - lower[-N_FREE_GATES:]
    )

    def _best(indices):
        if len(indices) == 0:
            return None
        return float(np.min(totals[indices]))

    lines = [
        "training set is the latin sample of junction, trajectory, and waveform",
        force_line(simulate_kwargs),
        stray_field_line(simulate_kwargs),
        heating_line(simulate_kwargs),
        "start and end height: 50 um",
        "bend-control heights: searched, within 15 um of 50 um",
        "free waveform gates: {}".format(N_FREE_GATES),
        "training designs: {}".format(int(n_train)),
        "latin varied designs: {}".format(len(latin)),
        "junction unit std: {:.3f}".format(float(unit[:, :8].std())),
        "trajectory unit std: {:.3f}".format(float(unit[:, 8:].std())),
        "waveform unit std: {:.3f}".format(float(gate_unit.std())),
        "nominal total quanta: {:.6g}".format(float(totals[0])),
        "reference waveform quanta: {:.6g}".format(float(totals[1]) if n_ref > 1 else float("nan")),
        "best latin total quanta: {:.6g}".format(_best(latin)),
        "CEM rounds: {}".format(int(iterations)),
        "CEM simulations: {}".format(len(cem)),
        "best CEM total quanta: {}".format(
            "{:.6g}".format(_best(cem)) if len(cem) else "none"
        ),
        "active rounds: {}".format(int(active_rounds)),
        "active simulations: {}".format(len(active)),
        "best active total quanta: {}".format(
            "{:.6g}".format(_best(active)) if len(active) else "none"
        ),
        "designs simulated: {}".format(len(totals)),
        "best additional total quanta: {}".format(
            "{:.6g}".format(_best(added)) if len(added) else "none"
        ),
        "best additional quanta: {}".format(
            " ".join("{:.6g}".format(float(v)) for v in quanta[plot_i].reshape(-1))
            if len(added) else "none"
        ),
        "best total quanta, including the training set: {:.6g}".format(float(np.min(totals))),
        "plotted design origin: {}".format(str(origin[plot_i])),
        "junction shift from nominal, unit norm: {:.4g}".format(junction_shift),
        "trajectory shift from nominal, unit norm: {:.4g}".format(path_shift),
        "waveform shift from nominal, gate norm: {:.4g}".format(gate_shift),
        "plotted bend radius s: {:.2f} um".format(design.cross_s * L_SCALE * 1e6),
        "plotted bend height: {:.2f} to {:.2f} um".format(
            design.interior_z.min() * L_SCALE * 1e6,
            design.interior_z.max() * L_SCALE * 1e6,
        ),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return {"best": float(np.min(totals)), "nominal": float(totals[0])}
