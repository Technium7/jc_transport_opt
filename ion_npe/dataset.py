"""Saved sets of simulated designs."""

from pathlib import Path

import numpy as np

from .design import DesignSpace, within_transport_caps
from .simulate import simulate_vectors, total_heating
from .trajectory import controls_stay_ordered, height_within_cap


def design_is_admissible(design):
    """Caps on crossing and height, and a control polygon that does not fold."""
    return (
        within_transport_caps(design)
        and controls_stay_ordered(design)
        and height_within_cap(design)
    )


class Dataset:
    """Design vectors and the final quanta of each noise seed."""

    def __init__(self, x, quanta, quanta_ideal, length_m, duration_s, seeds, space=None):
        self.x = np.asarray(x, dtype=float)
        self.quanta = np.asarray(quanta, dtype=float)
        self.quanta_ideal = np.asarray(quanta_ideal, dtype=float)
        self.length_m = np.asarray(length_m, dtype=float)
        self.duration_s = np.asarray(duration_s, dtype=float)
        self.seeds = np.asarray(seeds, dtype=int)
        self.space = space if space is not None else DesignSpace()
        if self.x.ndim != 2:
            raise ValueError("x must have shape (N, D)")
        if self.quanta.shape != (len(self.x), len(self.seeds), 3):
            raise ValueError("quanta must have shape (N, seeds, 3)")

    def __len__(self):
        return len(self.x)

    def successful(self):
        return np.isfinite(self.quanta).all(axis=(1, 2))

    def total_mean(self):
        totals = np.full(len(self), np.inf)
        mask = self.successful()
        if np.any(mask):
            totals[mask] = total_heating(self.quanta[mask])
        return totals

    def extended(self, x, quanta, length_m, duration_s):
        """Return a new dataset with simulated rows appended."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        quanta = np.asarray(quanta, dtype=float)
        if quanta.ndim == 2:
            quanta = quanta[None, :, :]
        length_m = np.atleast_1d(np.asarray(length_m, dtype=float))
        duration_s = np.atleast_1d(np.asarray(duration_s, dtype=float))
        return Dataset(
            np.vstack([self.x, x]),
            np.concatenate([self.quanta, quanta], axis=0),
            np.concatenate([self.quanta_ideal, np.full_like(quanta, np.nan)], axis=0),
            np.concatenate([self.length_m, length_m]),
            np.concatenate([self.duration_s, duration_s]),
            self.seeds,
            self.space,
        )

    def best_indices(self, k):
        totals = self.total_mean()
        k = min(int(k), int(np.isfinite(totals).sum()))
        if k <= 0:
            return np.array([], dtype=int)
        return np.argsort(totals)[:k]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            x=self.x,
            quanta=self.quanta,
            quanta_ideal=self.quanta_ideal,
            length_m=self.length_m,
            duration_s=self.duration_s,
            seeds=self.seeds,
            names=np.array(self.space.names),
            lower=self.space.lower,
            upper=self.space.upper,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        space = DesignSpace()
        if data["names"].shape != (space.dim,) or not np.array_equal(data["lower"], space.lower):
            # The rows are still usable. Bounds stay the current design space
            # so later sampling matches the code that will optimize them.
            pass
        return cls(
            data["x"],
            data["quanta"],
            data["quanta_ideal"],
            data["length_m"],
            data["duration_s"],
            data["seeds"],
            space,
        )


def _draw_proposal(space, rng, cover, fraction, cap_fraction, queue, remaining):
    """Next design vector. ``queue`` holds a Latin-hypercube batch."""
    if cover == "latin":
        if not queue:
            queue.extend(space.sample_latin(
                rng, max(int(remaining) * 2, 8),
                fraction=fraction, cap_fraction=cap_fraction,
            ))
        return queue.pop(0)
    return space.sample(rng, 1, fraction=fraction, cap_fraction=cap_fraction)[0]


def collect_dataset(
    n,
    seeds=(0, 1, 2, 3),
    fraction=0.5,
    cap_fraction=1.0,
    seed=0,
    space=None,
    max_attempts_factor=12,
    design="random",
    workers=1,
    **simulate_kwargs,
):
    """Simulate designs until ``n`` of them confine and finish.

    ``design="latin"`` places one sample in each equal bin of every coordinate
    so the range is covered evenly. ``design="random"`` draws uniformly.
    A design is skipped when the crossing is farther than 10 um, the ion
    leaves the 50 um height by more than 10 um, the bend polygon folds, or
    the RF curvature does not confine.

    ``workers`` above 1 simulates that many admissible designs at once.
    A ``scripts/mehta.sbatch`` step uses its task count instead.
    Successes are kept in sample order, so the saved rows match a serial run
    with the same seed.
    """
    from .parallel import active_workers, in_step

    if design not in ("random", "latin"):
        raise ValueError("design must be random or latin")
    space = space if space is not None else DesignSpace()
    seeds = [int(item) for item in seeds]
    workers = max(int(workers), 1)
    rng = np.random.default_rng(seed)
    rows = []
    quanta = []
    ideal = []
    lengths = []
    durations = []
    attempts = 0
    limit = max(int(n) * int(max_attempts_factor), int(n))
    queue = []
    width = active_workers(workers)
    print(
        "collecting {} designs, {} noise seeds, {} cover, fraction {:.2f}, caps 10 um, workers {}".format(
            n, len(seeds), design, fraction, width
        ),
        flush=True,
    )
    while len(rows) < n and attempts < limit:
        batch = []
        batch_attempts = []
        if in_step():
            batch_size = min(width * 4, n - len(rows))
        else:
            batch_size = 1 if width == 1 else min(width, n - len(rows))
        while len(batch) < batch_size and attempts < limit:
            proposal = _draw_proposal(
                space, rng, design, fraction, cap_fraction, queue, n - len(rows)
            )
            attempts += 1
            unpacked = space.unpack(proposal)
            if not design_is_admissible(unpacked):
                print(
                    "  skip {:3d}: outside the 10 um caps or the bend folds".format(attempts),
                    flush=True,
                )
                continue
            batch.append(proposal)
            batch_attempts.append(attempts)
        if not batch:
            break
        results = simulate_vectors(batch, seeds, workers=workers, **simulate_kwargs)
        for proposal, attempt, result in zip(batch, batch_attempts, results):
            if len(rows) >= n:
                break
            if not result.ok:
                print("  skip {:3d}: {}".format(attempt, result.message), flush=True)
                continue
            rows.append(proposal)
            quanta.append(result.quanta)
            if np.isfinite(result.quanta_ideal).all():
                ideal.append(result.quanta_ideal)
            else:
                ideal.append(np.full_like(result.quanta, np.nan))
            lengths.append(result.length_m)
            durations.append(result.duration_s)
            heating = float(total_heating(result.quanta))
            print(
                "  {:3d}/{}  heating {:.4g}  ({:.1f} to {:.1f} MHz)".format(
                    len(rows), n, heating, result.f_trans_MHz[0], result.f_trans_MHz[1]
                ),
                flush=True,
            )
    if len(rows) < n:
        raise RuntimeError(
            "collected {} of {} designs after {} attempts".format(len(rows), n, attempts)
        )
    return Dataset(
        np.vstack(rows),
        np.stack(quanta),
        np.stack(ideal),
        np.asarray(lengths),
        np.asarray(durations),
        np.asarray(seeds),
        space,
    )
