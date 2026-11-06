"""Local joint search that starts at the nominal path.

Each round collects a large neighborhood of the current design. The neighborhood
has its own junction, trajectory, and waveform samples, plus designs that move
all three together, so a fit can see a waveform-only improvement and a joint
one. Any of the three can be updated or kept. The next round recenters on the
lowest simulated heating and on the uncertain low-heating region the network
proposes, and collects another full batch there.
"""

from pathlib import Path

import numpy as np

from .active import active_ranking
from .constants import L_SCALE
from .design import DesignSpace
from .joint import (
    N_GATES, _admissible, _bounds, _location, _shifts, _simulate_row, _unpack, project_joint_row,
)
from .simulate import simulation_settings, stray_field_line, total_heating
from .train import log_rmse, train_predictor

MIN_TRAIN = 1000


def blocks(dim):
    """Index groups of a joint vector: junction, trajectory, waveform."""
    if dim <= N_GATES + 8:
        raise ValueError("joint vector is too short")
    junction = np.arange(0, 8)
    waveform = np.arange(dim - N_GATES, dim)
    trajectory = np.arange(8, dim - N_GATES)
    return {
        "junction": junction,
        "trajectory": trajectory,
        "waveform": waveform,
        "joint": np.arange(dim),
    }


def neighborhood_half(space, lower, upper):
    """Half-width of the box around the nominal path.

    Gates cover the shifts that change the time law. Heights can leave 50 um.
    The other controls stay in a fraction of their allowed range.
    """
    span = upper - lower
    names = list(space.names) + ["g{}".format(i) for i in range(N_GATES)]
    half = np.empty(len(names))
    for index, name in enumerate(names):
        if name.startswith("g"):
            half[index] = 0.08
        elif name.endswith("_z"):
            half[index] = 0.40 * span[index]
        elif name in ("curve_a", "curve_b", "cross_s"):
            half[index] = 0.30 * span[index]
        else:
            half[index] = 0.20 * span[index]
    return half


def box_around(center, half, lower, upper):
    lo = np.maximum(center - half, lower)
    hi = np.minimum(center + half, upper)
    hi = np.maximum(hi, lo + 1e-8)
    return lo, hi


def embed(center, values, free):
    """Place ``values`` into the free coordinates of ``center``."""
    row = np.array(center, dtype=float, copy=True)
    row[np.asarray(free, dtype=int)] = np.asarray(values, dtype=float)
    return row


def _split_counts(n, joint_only=False):
    """Waveform, junction, and trajectory slices, plus a joint slice.

    ``joint_only`` puts every design in the joint slice, so the junction, the
    trajectory, and the waveform move together.
    """
    n = int(n)
    if joint_only:
        return {"waveform": 0, "junction": 0, "trajectory": 0, "joint": n}
    each = n // 4
    return {
        "waveform": each,
        "junction": each,
        "trajectory": each,
        "joint": n - 3 * each,
    }


def _latin_rows(rng, n, center, lo, hi, free):
    from scipy.stats.qmc import LatinHypercube

    free = np.asarray(free, dtype=int)
    if len(free) == 0 or int(n) <= 0:
        return np.zeros((0, len(center)))
    unit = LatinHypercube(d=len(free), seed=rng).random(int(n))
    values = lo[free] + unit * (hi[free] - lo[free])
    rows = np.repeat(np.asarray(center, dtype=float).reshape(1, -1), int(n), axis=0)
    rows[:, free] = values
    return rows


def _collect_slice(rng, n, center, lo, hi, free, space, simulate_kwargs, label):
    rows = []
    quanta = []
    totals = []
    target = int(n)
    if target <= 0:
        return rows, quanta, totals
    tries = 0
    cap = max(target * 30, 30)
    while len(rows) < target and tries < cap:
        tries += 1
        batch = _latin_rows(rng, max((target - len(rows)) * 2, 16), center, lo, hi, free)
        for row in batch:
            if len(rows) >= target:
                break
            prepared = project_joint_row(row)
            result = _simulate_row(space, prepared, simulate_kwargs)
            if result is None:
                continue
            heating = float(total_heating(result.quanta))
            rows.append(prepared)
            quanta.append(np.asarray(result.quanta, dtype=float))
            totals.append(heating)
            if len(rows) == 1 or len(rows) % 50 == 0 or len(rows) == target:
                print(
                    "  {} {:4d}/{}  heating {:.4g}".format(label, len(rows), target, heating),
                    flush=True,
                )
    if len(rows) < target:
        raise RuntimeError("could not simulate {} {} designs".format(target, label))
    return rows, quanta, totals


def _stack(rows, quanta, totals, origin, new_rows, new_q, new_total, tag):
    if not new_rows:
        return rows, quanta, totals, origin
    stacked = [np.asarray(q, dtype=float).reshape(1, -1, 3) for q in new_q]
    return (
        np.vstack([rows, np.vstack(new_rows)]),
        np.concatenate([quanta, np.concatenate(stacked, axis=0)], axis=0),
        np.concatenate([totals, np.asarray(new_total, dtype=float)]),
        np.concatenate([origin, np.full(len(new_rows), tag, dtype="U24")]),
    )


def _save(path, rows, quanta, totals, origin, next_round, center, proposal_round=-1,
          proposal_center=None, proposal_half=None, active_center=None, active_half=None,
          seeds=None):
    if seeds is None:
        raise ValueError("noise seeds were not recorded")
    center = np.asarray(center, dtype=float)
    np.savez(
        path,
        rows=rows,
        quanta=quanta,
        totals=totals,
        seeds=np.asarray(seeds, dtype=int),
        origin=np.asarray(origin, dtype="U24"),
        next_round=np.int64(next_round),
        center=center,
        proposal_round=np.int64(proposal_round),
        proposal_center=center if proposal_center is None else np.asarray(proposal_center, dtype=float),
        proposal_half=np.zeros_like(center) if proposal_half is None else np.asarray(proposal_half, dtype=float),
        active_center=center if active_center is None else np.asarray(active_center, dtype=float),
        active_half=np.zeros_like(center) if active_half is None else np.asarray(active_half, dtype=float),
    )


def _load(path):
    data = np.load(path)
    center = np.asarray(data["center"], dtype=float)
    proposal_round = int(np.asarray(data["proposal_round"])) if "proposal_round" in data.files else -1
    quanta = np.asarray(data["quanta"], dtype=float)
    if "seeds" in data.files:
        saved_seeds = np.asarray(data["seeds"], dtype=int)
    else:
        n_saved = quanta.shape[1] if quanta.ndim == 3 else 1
        saved_seeds = np.arange(n_saved)
    return (
        np.asarray(data["rows"], dtype=float),
        quanta,
        np.asarray(data["totals"], dtype=float),
        np.asarray(data["origin"]).astype(str),
        int(np.asarray(data["next_round"])),
        center,
        proposal_round,
        np.asarray(data["proposal_center"], dtype=float) if "proposal_center" in data.files else center,
        np.asarray(data["proposal_half"], dtype=float) if "proposal_half" in data.files else np.zeros_like(center),
        np.asarray(data["active_center"], dtype=float) if "active_center" in data.files else center,
        np.asarray(data["active_half"], dtype=float) if "active_half" in data.files else np.zeros_like(center),
        saved_seeds,
    )


def _remaining(origin, tag, count):
    return max(int(count) - int(np.sum(np.asarray(origin) == tag)), 0)


def _collect_neighborhood(
    rng, counts, center, half, lower, upper, space, simulate_kwargs, tag,
    rows, quanta, totals, origin, on_update=None,
):
    groups = blocks(len(center))
    lo, hi = box_around(center, half, lower, upper)
    for name in ("waveform", "junction", "trajectory", "joint"):
        label = "{}-{}".format(tag, name)
        need = _remaining(origin, label, counts[name])
        if need <= 0:
            continue
        got_rows, got_q, got_total = _collect_slice(
            rng, need, center, lo, hi, groups[name], space, simulate_kwargs, label,
        )
        rows, quanta, totals, origin = _stack(
            rows, quanta, totals, origin, got_rows, got_q, got_total, label,
        )
        if on_update is not None:
            on_update(rows, quanta, totals, origin)
    return rows, quanta, totals, origin


def _fit(rows, quanta, epochs, lbfgs_maxiter, seed):
    if len(rows) < MIN_TRAIN:
        raise RuntimeError(
            "training step has {} designs; at least {} are required".format(len(rows), MIN_TRAIN)
        )
    predictor, _history = train_predictor(
        rows, quanta, kind="stochastic", hidden=48, epochs=epochs, seed=seed,
        val_fraction=0.2, lbfgs_maxiter=lbfgs_maxiter, log_every=20, batch_size=64,
    )
    score = log_rmse(predictor, rows, quanta, n_samples=16, seed=seed + 9)
    print("  training log10 RMSE on the collected designs {:.3f}".format(score), flush=True)
    return predictor


def _next_half(rows, totals, initial_half):
    """Box around the best design, wide enough to cover the other low-heating ones."""
    order = np.argsort(totals)
    elite_n = int(min(64, max(32, len(totals) // 20)))
    elite = rows[order[:elite_n]]
    best = rows[int(order[0])]
    spread = np.maximum(1.5 * elite.std(axis=0), 0.40 * initial_half)
    half = np.minimum(spread, initial_half)
    half[-N_GATES:] = np.clip(half[-N_GATES:], 0.04, initial_half[-N_GATES:])
    return best, half


def _active_center(predictor, rng, center, half, lower, upper, space):
    """Center of the uncertain low-heating part of the current neighborhood."""
    lo, hi = box_around(center, half, lower, upper)
    pool = _latin_rows(rng, 512, center, lo, hi, np.arange(len(center)))
    keep = []
    for row in pool:
        design = _unpack(space, row)
        if _admissible(design):
            keep.append(row)
        if len(keep) == 256:
            break
    if len(keep) < 32:
        return center, 0.5 * half
    pool = np.vstack(keep)
    predicted, log_std = _location(predictor, pool, rng)
    order = active_ranking(predicted, 10.0 ** log_std, phonon_quantile=0.5)
    chosen = pool[order[:32]]
    return chosen.mean(axis=0), np.maximum(chosen.std(axis=0), 0.35 * half)


def _slice_bests(origin, totals):
    parts = []
    for suffix in ("waveform", "junction", "trajectory", "joint", "active"):
        idx = np.array([i for i, tag in enumerate(origin) if str(tag).endswith(suffix)])
        if len(idx) == 0:
            continue
        parts.append("{} {:.4g} (n={})".format(suffix, float(np.min(totals[idx])), len(idx)))
    return ", ".join(parts)


def _warm_row(path):
    data = np.load(path)
    rows = np.asarray(data["rows"], dtype=float)
    totals = np.asarray(data["totals"], dtype=float)
    return project_joint_row(rows[int(np.argmin(totals))])


def optimize_local(
    out,
    n=2000,
    per_round=1000,
    iterations=3,
    epochs=40,
    lbfgs_maxiter=15,
    seed=0,
    simulate_kwargs=None,
    cooptimize=False,
    warm_start=None,
):
    """Start at the nominal path and iteratively collect nearby joint designs."""
    if int(n) < MIN_TRAIN:
        raise ValueError("the first collection needs at least {} designs".format(MIN_TRAIN))
    if int(per_round) < MIN_TRAIN:
        raise ValueError("each optimization step needs at least {} new designs".format(MIN_TRAIN))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    simulate_kwargs = simulation_settings(simulate_kwargs)
    seeds = simulate_kwargs["seeds"]
    space = DesignSpace()
    lower, upper = _bounds(space)
    nominal = np.concatenate([space.nominal(), np.zeros(N_GATES)])
    initial_half = neighborhood_half(space, lower, upper)
    rng = np.random.default_rng(seed)
    cache = out / "local.npz"

    proposal_round = -1
    proposal_center = nominal.copy()
    proposal_half = initial_half.copy()
    active_center = nominal.copy()
    active_half = initial_half.copy()
    if cache.exists():
        (
            rows, quanta, totals, origin, next_round, center, proposal_round,
            proposal_center, proposal_half, active_center, active_half, saved_seeds,
        ) = _load(cache)
        if tuple(int(item) for item in saved_seeds) != tuple(seeds):
            raise RuntimeError(
                "saved designs used noise seeds {}; this run uses {}".format(
                    tuple(int(item) for item in saved_seeds), tuple(seeds)
                )
            )
        print(
            "resuming with {} designs, next round {}".format(len(rows), next_round),
            flush=True,
        )
    else:
        print("nominal path, nominal waveform", flush=True)
        print("  {}".format(stray_field_line(simulate_kwargs)), flush=True)
        nominal_result = _simulate_row(space, nominal, simulate_kwargs)
        if nominal_result is None:
            raise RuntimeError("nominal design is not admissible")
        heating = float(total_heating(nominal_result.quanta))
        print("  heating {:.4g}".format(heating), flush=True)
        rows = nominal.reshape(1, -1)
        quanta = nominal_result.quanta.reshape(1, *nominal_result.quanta.shape)
        totals = np.asarray([heating], dtype=float)
        origin = np.array(["nominal"], dtype="U24")
        center = nominal.copy()
        next_round = 0
        _save(cache, rows, quanta, totals, origin, next_round, center, seeds=seeds)

    if next_round == 0 and warm_start and not np.any(origin == "reference"):
        print("reference design from {}".format(warm_start), flush=True)
        reference = _warm_row(warm_start)
        reference_result = _simulate_row(space, reference, simulate_kwargs)
        if reference_result is None:
            raise RuntimeError("the warm-start design is not admissible")
        heating = float(total_heating(reference_result.quanta))
        print("  heating {:.4g}".format(heating), flush=True)
        rows, quanta, totals, origin = _stack(
            rows, quanta, totals, origin, [reference], [reference_result.quanta], [heating], "reference",
        )
        center = reference
        _save(cache, rows, quanta, totals, origin, next_round, center, seeds=seeds)

    if next_round == 0:
        where = "the warm-start design" if warm_start else "the nominal path"
        print(
            "round 0  {} designs around {} ({})".format(
                int(n),
                where,
                "junction, trajectory, and waveform, each updated or kept",
            ),
            flush=True,
        )
        def _checkpoint(rows, quanta, totals, origin):
            _save(cache, rows, quanta, totals, origin, 0, center, seeds=seeds)

        rows, quanta, totals, origin = _collect_neighborhood(
            rng, _split_counts(n), center, initial_half, lower, upper, space,
            simulate_kwargs, "r0", rows, quanta, totals, origin, on_update=_checkpoint,
        )
        next_round = 1
        center = nominal.copy()
        _save(cache, rows, quanta, totals, origin, next_round, center, seeds=seeds)
        print(
            "  collected {}  best {:.4g}  {}".format(
                len(rows), float(np.min(totals)), _slice_bests(origin, totals)
            ),
            flush=True,
        )

    predictor = None
    for step in range(next_round, int(iterations) + 1):
        if proposal_round != step:
            print("training on {} designs before round {}".format(len(rows), step), flush=True)
            predictor = _fit(rows, quanta, epochs, lbfgs_maxiter, seed + step)
            center, proposal_half = _next_half(rows, totals, initial_half)
            proposal_center = center
            active_center, active_half = _active_center(
                predictor, rng, center, proposal_half, lower, upper, space,
            )
            proposal_round = step
            _save(
                cache, rows, quanta, totals, origin, step, center, proposal_round,
                proposal_center, proposal_half, active_center, active_half, seeds=seeds,
            )
        print(
            "round {}  best so far {:.4g}  collecting {} nearby and in the uncertain region".format(
                step, float(np.min(totals)), int(per_round)
            ),
            flush=True,
        )
        main_n = int(per_round) * 3 // 4
        active_n = int(per_round) - main_n

        def _checkpoint(rows, quanta, totals, origin, step=step):
            _save(
                cache, rows, quanta, totals, origin, step, proposal_center, step,
                proposal_center, proposal_half, active_center, active_half, seeds=seeds,
            )

        rows, quanta, totals, origin = _collect_neighborhood(
            rng, _split_counts(main_n), proposal_center, proposal_half, lower, upper, space,
            simulate_kwargs, "r{}".format(step), rows, quanta, totals, origin, on_update=_checkpoint,
        )
        label = "r{}-active".format(step)
        need = _remaining(origin, label, active_n)
        if need > 0:
            got_rows, got_q, got_total = _collect_slice(
                rng, need, active_center, *box_around(active_center, active_half, lower, upper),
                np.arange(len(proposal_center)), space, simulate_kwargs, label,
            )
            rows, quanta, totals, origin = _stack(
                rows, quanta, totals, origin, got_rows, got_q, got_total, label,
            )
        next_round = step + 1
        proposal_round = -1
        _save(cache, rows, quanta, totals, origin, next_round, proposal_center, seeds=seeds)
        print(
            "  collected {}  best {:.4g}  {}".format(
                len(rows), float(np.min(totals)), _slice_bests(origin, totals)
            ),
            flush=True,
        )

    print("training on {} designs".format(len(rows)), flush=True)
    predictor = _fit(rows, quanta, epochs, lbfgs_maxiter, seed + 50)
    predictor.save(out / "stochastic.npz")

    best_i = int(np.argmin(totals))
    design = _unpack(space, rows[best_i])
    junction_shift, path_shift, gate_shift = _shifts(rows[best_i], nominal, space)
    from .plotting import plot_best_design, plot_trajectory

    plot_trajectory(design, None, out / "best_path.png")
    plot_best_design(
        design, out / "best_design.png",
        reference=space.nominal_design(), total_quanta=float(totals[best_i]),
    )

    def _best_tag(suffix):
        idx = np.array([i for i, tag in enumerate(origin) if tag.endswith(suffix)])
        if len(idx) == 0:
            return None, 0
        return float(np.min(totals[idx])), len(idx)

    wave_best, wave_n = _best_tag("waveform")
    junc_best, junc_n = _best_tag("junction")
    traj_best, traj_n = _best_tag("trajectory")
    joint_best, joint_n = _best_tag("joint")
    active_best, active_n = _best_tag("active")
    lines = [
        "junction, trajectory, and waveform, each updated or kept"
        if cooptimize else "local search starting at the nominal path",
        stray_field_line(simulate_kwargs),
        "start and end height: 50 um",
        "designs simulated: {}".format(len(totals)),
        "waveform samples: {}".format(wave_n),
        "junction samples: {}".format(junc_n),
        "trajectory samples: {}".format(traj_n),
        "joint samples: {}".format(joint_n),
        "active-region samples: {}".format(active_n),
        "nominal total quanta: {:.6g}".format(float(totals[0])),
        "best waveform-only total quanta: {}".format(
            "{:.6g}".format(wave_best) if wave_best is not None else "none"
        ),
        "best junction-only total quanta: {}".format(
            "{:.6g}".format(junc_best) if junc_best is not None else "none"
        ),
        "best trajectory-only total quanta: {}".format(
            "{:.6g}".format(traj_best) if traj_best is not None else "none"
        ),
        "best joint total quanta: {}".format(
            "{:.6g}".format(joint_best) if joint_best is not None else "none"
        ),
        "best active-region total quanta: {}".format(
            "{:.6g}".format(active_best) if active_best is not None else "none"
        ),
        "best total quanta: {:.6g}".format(float(totals[best_i])),
        "best mean quanta: {}".format(
            " ".join("{:.6g}".format(float(v)) for v in np.mean(quanta[best_i], axis=0))
        ),
        "best total std across samples: {:.6g}".format(
            float(np.std(np.sum(quanta[best_i], axis=-1)))
        ),
        "best origin: {}".format(str(origin[best_i])),
        "junction shift from nominal, unit norm: {:.4g}".format(junction_shift),
        "trajectory shift from nominal, unit norm: {:.4g}".format(path_shift),
        "waveform shift from nominal, gate norm: {:.4g}".format(gate_shift),
        "best crossing distance: {:.2f} um".format(design.cross_s * L_SCALE * 1e6),
        "best crossing height: {:.2f} um".format(design.z_cross * L_SCALE * 1e6),
        "best bend entrance: {:.2f} um".format(design.z_bend * L_SCALE * 1e6),
        "best arm height: {:.2f} to {:.2f} um".format(
            design.arm_xyz[:, 2].min() * L_SCALE * 1e6,
            design.arm_xyz[:, 2].max() * L_SCALE * 1e6,
        ),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return {"best": float(totals[best_i]), "nominal": float(totals[0])}
