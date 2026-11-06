"""Split simulation batches across the tasks of one SLURM step.

Submit with ``scripts/mehta.sbatch``. That step is 12 tasks on the mehta
node, 48 GB, 8 hours. Rank 0 runs the command and trains. Every task,
including rank 0, simulates the rows whose index modulo the task count
equals its rank. The other tasks wait until the next batch or a stop mark.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


def rank_and_size():
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = int(os.environ.get("SLURM_NTASKS", "1"))
    return rank, max(size, 1)


def in_step():
    """True when this process is one task of a multi-task SLURM step."""
    return rank_and_size()[1] > 1


def active_workers(requested=1):
    """Task count inside a parallel step, otherwise the requested pool size."""
    _rank, size = rank_and_size()
    if size > 1:
        return size
    return max(int(requested), 1)


def work_dir():
    return Path(os.environ.get("ION_NPE_FARM", "outputs/farm"))


def task_indices(n, rank, size):
    """Row indices owned by ``rank`` when ``size`` tasks split the batch."""
    return list(range(int(rank), int(n), int(size)))


def merge_parts(parts, n):
    """Place each task's successful quanta back in batch order.

    ``parts`` holds ``(indices, ok, quanta)``. A failed row stays ``None``.
    """
    merged = [None] * int(n)
    for indices, ok, quanta in parts:
        for slot, index in enumerate(np.asarray(indices, dtype=int)):
            if bool(ok[slot]):
                merged[int(index)] = np.asarray(quanta[slot], dtype=float)
    return merged


def guard(run):
    """Run ``run`` on rank 0. Other tasks simulate until rank 0 stops."""
    rank, size = rank_and_size()
    if size > 1 and rank != 0:
        serve()
        return None
    try:
        return run()
    finally:
        if size > 1 and rank == 0:
            stop()


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _pack_kwargs(simulate_kwargs):
    return {key: _jsonable(value) for key, value in dict(simulate_kwargs).items()}


def _unpack_kwargs(packed):
    data = dict(packed)
    if "seeds" in data:
        data["seeds"] = tuple(int(seed) for seed in data["seeds"])
    return data


def _publish(directory, name, text):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    temporary = directory / (name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, target)


def _read_mark(directory, name):
    path = directory / name
    if not path.exists():
        return ""
    return path.read_text().strip()


def _save_share(path, indices, ok, quanta, ideal, length_m, duration_s, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            indices=np.asarray(indices, dtype=int),
            ok=np.asarray(ok, dtype=bool),
            quanta=np.asarray(quanta, dtype=float),
            quanta_ideal=np.asarray(ideal, dtype=float),
            length_m=np.asarray(length_m, dtype=float),
            duration_s=np.asarray(duration_s, dtype=float),
            message=np.asarray(message, dtype="U256"),
        )
    os.replace(temporary, path)


def _load_share(path):
    with np.load(path) as data:
        return {
            "indices": np.asarray(data["indices"], dtype=int),
            "ok": np.asarray(data["ok"], dtype=bool),
            "quanta": np.asarray(data["quanta"], dtype=float),
            "quanta_ideal": np.asarray(data["quanta_ideal"], dtype=float),
            "length_m": np.asarray(data["length_m"], dtype=float),
            "duration_s": np.asarray(data["duration_s"], dtype=float),
            "message": np.asarray(data["message"]).astype(str),
        }


def _blank(n_seeds):
    quanta = np.full((n_seeds, 3), np.nan)
    return False, quanta, quanta.copy(), np.nan, np.nan, ""


def _record(result, n_seeds):
    """Pull the fields later training and search need off one run."""
    if result is None:
        return _blank(n_seeds)
    quanta = np.full((n_seeds, 3), np.nan)
    ideal = quanta.copy()
    got = np.asarray(result.quanta, dtype=float)
    if got.shape == (n_seeds, 3):
        quanta = got
    ideal_got = getattr(result, "quanta_ideal", None)
    if ideal_got is not None and np.asarray(ideal_got).shape == (n_seeds, 3):
        ideal = np.asarray(ideal_got, dtype=float)
    return (
        bool(getattr(result, "ok", True)),
        quanta,
        ideal,
        float(getattr(result, "length_m", np.nan)),
        float(getattr(result, "duration_s", np.nan)),
        str(getattr(result, "message", "")),
    )


def _simulate_joint_row(row, simulate_kwargs):
    from .design import DesignSpace
    from .joint import _simulate_row

    try:
        return _simulate_row(DesignSpace(), np.asarray(row, dtype=float), simulate_kwargs)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as err:
        print("  skip: {}".format(err), flush=True)
        return None


def _simulate_design_row(row, seeds, simulate_kwargs):
    from .design import DesignSpace
    from .simulate import simulate_design

    kwargs = dict(simulate_kwargs)
    kwargs.pop("seeds", None)
    try:
        return simulate_design(
            DesignSpace().unpack(np.asarray(row, dtype=float)),
            seeds=tuple(int(seed) for seed in seeds),
            **kwargs,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as err:
        print("  skip: {}".format(err), flush=True)
        return None


def _evaluate(kind, row, seeds, simulate_kwargs):
    if kind == "joint":
        return _simulate_joint_row(row, simulate_kwargs)
    if kind == "design":
        return _simulate_design_row(row, seeds, simulate_kwargs)
    raise ValueError("unknown simulation kind {}".format(kind))


def _simulate_share(rows, rank, size, kind, seeds, simulate_kwargs):
    indices = task_indices(len(rows), rank, size)
    n_seeds = max(len(seeds), 1)
    quanta = np.full((len(indices), n_seeds, 3), np.nan)
    ideal = np.full((len(indices), n_seeds, 3), np.nan)
    length_m = np.full(len(indices), np.nan)
    duration_s = np.full(len(indices), np.nan)
    message = np.empty(len(indices), dtype="U256")
    ok = np.zeros(len(indices), dtype=bool)
    for slot, index in enumerate(indices):
        got_ok, got_q, got_ideal, got_length, got_duration, got_message = _record(
            _evaluate(kind, rows[index], seeds, simulate_kwargs), n_seeds
        )
        ok[slot] = got_ok
        quanta[slot] = got_q
        ideal[slot] = got_ideal
        length_m[slot] = got_length
        duration_s[slot] = got_duration
        message[slot] = got_message
    return indices, ok, quanta, ideal, length_m, duration_s, message


def _run_batch(directory, batch, rank, size):
    batch_dir = directory / "batch_{}".format(batch)
    rows = np.load(batch_dir / "rows.npy")
    spec = json.loads((batch_dir / "spec.json").read_text())
    seeds = tuple(int(seed) for seed in spec["seeds"])
    simulate_kwargs = _unpack_kwargs(spec["kwargs"])
    share = _simulate_share(rows, rank, size, spec["kind"], seeds, simulate_kwargs)
    _save_share(batch_dir / "rank_{}.npz".format(rank), *share)


class _Quanta:
    """Successful joint run. Callers read ``quanta``."""

    def __init__(self, quanta):
        self.ok = True
        self.quanta = np.asarray(quanta, dtype=float)


def _as_run(ok, quanta, ideal, length_m, duration_s, message):
    from .simulate import RunResult

    blank = np.full_like(quanta, np.nan)
    return RunResult(
        ok=bool(ok),
        quanta=np.asarray(quanta, dtype=float),
        quanta_ideal=np.asarray(ideal, dtype=float),
        peak_quanta=blank,
        noise_max_Vm=np.zeros_like(quanta),
        message=str(message),
        duration_s=float(duration_s),
        length_m=float(length_m),
        f_trans_MHz=(np.nan, np.nan),
        z_range=(np.nan, np.nan),
        n_hold=0,
    )


def _scatter(rows, simulate_kwargs, kind, seeds):
    rank, size = rank_and_size()
    directory = work_dir()
    directory.mkdir(parents=True, exist_ok=True)
    current = _read_mark(directory, "batch")
    batch = int(current) + 1 if current.isdigit() else 1
    batch_dir = directory / "batch_{}".format(batch)
    batch_dir.mkdir(parents=True, exist_ok=True)
    np.save(batch_dir / "rows.npy", np.vstack([np.asarray(row, dtype=float) for row in rows]))
    spec = {
        "kind": kind,
        "seeds": [int(seed) for seed in seeds],
        "kwargs": _pack_kwargs(simulate_kwargs),
    }
    (batch_dir / "spec.json").write_text(json.dumps(spec))
    print("  {} tasks collecting {} designs".format(size, len(rows)), flush=True)
    _publish(directory, "batch", str(batch))
    _run_batch(directory, batch, rank, size)
    parts = []
    for other in range(size):
        path = batch_dir / "rank_{}.npz".format(other)
        while not path.exists():
            if _read_mark(directory, "batch") == "stop":
                raise RuntimeError("collection stopped while waiting for rank {}".format(other))
            time.sleep(0.2)
        parts.append(_load_share(path))
    return parts


def _merged_runs(parts, n):
    runs = [None] * int(n)
    for part in parts:
        for slot, index in enumerate(part["indices"]):
            runs[int(index)] = _as_run(
                part["ok"][slot],
                part["quanta"][slot],
                part["quanta_ideal"][slot],
                part["length_m"][slot],
                part["duration_s"][slot],
                part["message"][slot],
            )
    return runs


def map_joint(rows, simulate_kwargs):
    """Simulate joint rows in order. Failed rows are ``None``."""
    rows = [np.asarray(row, dtype=float) for row in rows]
    if len(rows) == 0:
        return []
    seeds = tuple(int(seed) for seed in simulate_kwargs.get("seeds", (0,)))
    if not in_step() or len(rows) == 1:
        out = []
        for row in rows:
            result = _simulate_joint_row(row, simulate_kwargs)
            out.append(None if result is None or not getattr(result, "ok", True) else result)
        return out
    runs = _merged_runs(_scatter(rows, simulate_kwargs, "joint", seeds), len(rows))
    return [None if run is None or not run.ok else _Quanta(run.quanta) for run in runs]


def map_designs(vectors, seeds, simulate_kwargs):
    """Simulate design vectors in order. Each entry is a ``RunResult``."""
    vectors = [np.asarray(vector, dtype=float) for vector in vectors]
    seeds = tuple(int(seed) for seed in seeds)
    if len(vectors) == 0:
        return []
    if not in_step() or len(vectors) == 1:
        runs = [_simulate_design_row(vector, seeds, simulate_kwargs) for vector in vectors]
        return [run if run is not None else _as_run(False, *_blank(len(seeds))[1:]) for run in runs]
    runs = _merged_runs(_scatter(vectors, simulate_kwargs, "design", seeds), len(vectors))
    return [run if run is not None else _as_run(False, *_blank(len(seeds))[1:]) for run in runs]


def serve():
    """Wait for rank 0 to publish batches. Return when it publishes stop."""
    directory = work_dir()
    directory.mkdir(parents=True, exist_ok=True)
    rank, size = rank_and_size()
    seen = 0
    while True:
        mark = _read_mark(directory, "batch")
        if mark == "stop":
            return
        if mark.isdigit() and int(mark) != seen:
            seen = int(mark)
            _run_batch(directory, seen, rank, size)
            continue
        time.sleep(0.2)


def stop():
    """Tell every waiting task that the command is finished."""
    _publish(work_dir(), "batch", "stop")
