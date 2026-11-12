"""Split ion simulations across the nodes of one Slurm step.

Rank 0 runs collection, training, and the search. Every rank, including rank
0, integrates its share of a batch. Batches are files in a directory shared by
the nodes, so the job does not need MPI. One process, or a Slurm step with one
task, keeps the original one-at-a-time loop.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


_SIM_KEYS = (
    "seeds",
    "speed_m_s",
    "voltage_update_s",
    "noise_std_Vm",
    "force",
    "with_ideal",
    "sim_dt",
    "n_geom",
    "heating",
)

_FARM = None


def width_for(need, size):
    """How many designs to simulate before the next synchronization.

    A single process keeps a width of one, so the sampled designs stay in the
    same order as the local loop. Several nodes take a wider batch, capped so
    a wave does not run past the designs this round still needs.
    """
    need = max(int(need), 1)
    if int(size) <= 1:
        return 1
    return min(need, max(int(size) * 8, 32))


def owned_indices(n, rank, size):
    """Row numbers this rank integrates. Rank 0 takes 0, size, 2*size, ..."""
    return list(range(int(rank), int(n), int(size)))


def get_farm():
    """The farm for this process. One object, so the step counter stays put."""
    global _FARM
    if _FARM is None:
        _FARM = Farm.from_environ()
    return _FARM


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _n_seeds(kwargs):
    seeds = list(kwargs.get("seeds", [0]))
    return max(len(seeds), 1)


def _sim_kwargs(kwargs):
    return {key: kwargs[key] for key in _SIM_KEYS if key in kwargs}


def _failure(message, n_seeds):
    nan = np.full((int(n_seeds), 3), np.nan)
    return {
        "ok": False,
        "quanta": nan,
        "quanta_ideal": nan.copy(),
        "length_m": np.nan,
        "duration_s": np.nan,
        "f_trans_MHz": (np.nan, np.nan),
        "message": str(message),
    }


def _pack(result, n_seeds):
    quanta = np.asarray(result.quanta, dtype=float)
    ideal = np.asarray(result.quanta_ideal, dtype=float)
    if quanta.shape != (n_seeds, 3):
        quanta = np.full((n_seeds, 3), np.nan)
    if ideal.shape != (n_seeds, 3):
        ideal = np.full((n_seeds, 3), np.nan)
    ok = bool(result.ok) and bool(np.isfinite(quanta).all())
    low, high = result.f_trans_MHz
    return {
        "ok": ok,
        "quanta": quanta,
        "quanta_ideal": ideal,
        "length_m": float(result.length_m),
        "duration_s": float(result.duration_s),
        "f_trans_MHz": (float(low), float(high)),
        "message": "" if ok else (result.message or "simulation failed"),
    }


def evaluate_row(kind, row, simulate_kwargs):
    """Integrate one saved design. ``echo`` is only for the farm tests."""
    row = np.asarray(row, dtype=float)
    n_seeds = _n_seeds(simulate_kwargs)
    if kind == "echo":
        quanta = np.full((n_seeds, 3), float(np.sum(row)))
        return {
            "ok": True,
            "quanta": quanta,
            "quanta_ideal": np.full_like(quanta, np.nan),
            "length_m": 0.0,
            "duration_s": 0.0,
            "f_trans_MHz": (0.0, 0.0),
            "message": "",
        }
    try:
        if kind == "design":
            from .dataset import design_is_admissible
            from .design import DesignSpace
            from .simulate import simulate_design

            design = DesignSpace().unpack(row)
            if not design_is_admissible(design):
                return _failure("outside the 15 um caps or the bend folds", n_seeds)
            result = simulate_design(design, **_sim_kwargs(simulate_kwargs))
        elif kind == "joint":
            from .design import DesignSpace
            from .joint import _admissible, _unpack
            from .simulate import simulate_design

            space = DesignSpace()
            design = _unpack(space, row)
            if not _admissible(design):
                return _failure("outside the 15 um caps or the waveform folds", n_seeds)
            result = simulate_design(design, **_sim_kwargs(simulate_kwargs))
        else:
            raise ValueError("unknown farm kind {}".format(kind))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as err:
        return _failure(str(err), n_seeds)
    return _pack(result, n_seeds)


def _atomic_write(path, write):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "wb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path, payload):
    text = json.dumps(payload).encode("utf-8")
    _atomic_write(path, lambda handle: handle.write(text))


def _write_array(path, array):
    def write(handle):
        np.save(handle, np.asarray(array))

    _atomic_write(path, write)


def _write_npz(path, **arrays):
    def write(handle):
        np.savez(handle, **arrays)

    _atomic_write(path, write)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_array(path):
    last = None
    for _ in range(20):
        try:
            return np.load(path)
        except (OSError, ValueError) as err:
            last = err
            time.sleep(0.1)
    raise RuntimeError("could not read {}: {}".format(path, last))


def _load_npz(path):
    last = None
    for _ in range(20):
        try:
            with np.load(path) as data:
                return {key: np.array(data[key], copy=True) for key in data.files}
        except (OSError, ValueError) as err:
            last = err
            time.sleep(0.1)
    raise RuntimeError("could not read {}: {}".format(path, last))


class Farm:
    """One Slurm step. Rank 0 submits batches; every rank integrates a slice."""

    def __init__(self, work, rank, size, poll_s=0.25, timeout_s=None):
        self.work = Path(work)
        self.rank = int(rank)
        self.size = max(int(size), 1)
        self.poll_s = float(poll_s)
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.step = 0
        self._announced = False

    @property
    def is_root(self):
        return self.rank == 0

    @classmethod
    def from_environ(cls):
        """Ranks come from ``srun``. A bare batch script is one process.

        Slurm sets ``SLURM_NTASKS`` on the batch script itself, before
        ``srun`` has started the other nodes. The farm stays local unless
        this process has a task rank, which ``srun`` provides.
        """
        if "SLURM_PROCID" not in os.environ:
            rank, size = 0, 1
        else:
            rank = int(os.environ.get("SLURM_PROCID", "0"))
            size = int(os.environ.get("SLURM_NTASKS", os.environ.get("SLURM_NPROCS", "1")))
        base = Path(os.environ.get("ION_FARM_DIR", "outputs/farm"))
        job = os.environ.get("SLURM_JOB_ID", "local")
        step = os.environ.get("SLURM_STEP_ID", "0")
        return cls(base / "{}_{}".format(job, step), rank, size)

    def simulate(self, kind, rows, simulate_kwargs):
        """Integrate every row. Results are in the same order as ``rows``."""
        rows = np.asarray(rows, dtype=float)
        if rows.size == 0:
            return []
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        kwargs = dict(simulate_kwargs or {})
        if self.size <= 1:
            return [evaluate_row(kind, row, kwargs) for row in rows]
        if not self.is_root:
            raise RuntimeError("only rank 0 submits a farm batch")
        self.work.mkdir(parents=True, exist_ok=True)
        if not self._announced:
            print(
                "simulations are split across {} nodes ({})".format(self.size, self.work),
                flush=True,
            )
            self._announced = True
        step = self.step
        command = {
            "op": "simulate",
            "kind": str(kind),
            "n": int(len(rows)),
            "kwargs": _plain(kwargs),
        }
        _write_array(self.work / "rows_{}.npy".format(step), rows)
        self._publish(step, command)
        self._run_shard(step, command, rows)
        self._wait_ready(step, range(self.size))
        merged = self._merge(step, len(rows))
        self.step += 1
        return merged

    def serve(self):
        """Non-root ranks. Block until rank 0 publishes a stop command."""
        if self.is_root:
            raise RuntimeError("rank 0 runs the search; it does not serve")
        self.work.mkdir(parents=True, exist_ok=True)
        step = 0
        while True:
            command = self._wait_command(step)
            if command["op"] == "stop":
                self._mark_ready(step)
                return
            if command["op"] != "simulate":
                raise RuntimeError("unknown farm command {}".format(command["op"]))
            rows = _load_array(self.work / "rows_{}.npy".format(step))
            self._run_shard(step, command, rows)
            step += 1

    def stop(self):
        """Tell the other ranks to exit. Safe to call when this process is alone."""
        if self.size <= 1 or not self.is_root:
            return
        step = self.step
        self.work.mkdir(parents=True, exist_ok=True)
        self._publish(step, {"op": "stop", "kind": "", "n": 0, "kwargs": {}})
        self._wait_ready(step, range(1, self.size))
        self.step += 1

    def _publish(self, step, command):
        _write_json(self.work / "cmd_{}.json".format(step), command)
        ready = self.work / "cmd_{}.ready".format(step)
        ready.write_text("ok", encoding="utf-8")

    def _wait_command(self, step):
        path = self.work / "cmd_{}.ready".format(step)
        self._wait_files([path])
        return _load_json(self.work / "cmd_{}.json".format(step))

    def _run_shard(self, step, command, rows):
        n_seeds = _n_seeds(command["kwargs"])
        indices = owned_indices(len(rows), self.rank, self.size)
        packed = []
        for index in indices:
            try:
                packed.append(evaluate_row(command["kind"], rows[index], command["kwargs"]))
            except Exception as err:
                packed.append(_failure("{}: {}".format(type(err).__name__, err), n_seeds))
        if indices:
            print(
                "  node {} simulated {}".format(self.rank, len(indices)),
                flush=True,
            )
        if packed:
            quanta = np.stack([item["quanta"] for item in packed])
            ideal = np.stack([item["quanta_ideal"] for item in packed])
            messages = np.asarray([item["message"] for item in packed], dtype="U400")
            frequencies = np.asarray([item["f_trans_MHz"] for item in packed], dtype=float)
            lengths = np.asarray([item["length_m"] for item in packed], dtype=float)
            durations = np.asarray([item["duration_s"] for item in packed], dtype=float)
            ok = np.asarray([item["ok"] for item in packed], dtype=bool)
        else:
            quanta = np.zeros((0, n_seeds, 3))
            ideal = np.zeros((0, n_seeds, 3))
            messages = np.asarray([], dtype="U400")
            frequencies = np.zeros((0, 2))
            lengths = np.zeros(0)
            durations = np.zeros(0)
            ok = np.zeros(0, dtype=bool)
        _write_npz(
            self.work / "shard_{}_{}.npz".format(step, self.rank),
            index=np.asarray(indices, dtype=int),
            ok=ok,
            quanta=quanta,
            quanta_ideal=ideal,
            length_m=lengths,
            duration_s=durations,
            frequency=frequencies,
            message=messages,
        )
        self._mark_ready(step)

    def _mark_ready(self, step):
        (self.work / "ready_{}_{}".format(step, self.rank)).write_text("ok", encoding="utf-8")

    def _wait_ready(self, step, ranks):
        paths = [self.work / "ready_{}_{}".format(step, rank) for rank in ranks]
        self._wait_files(paths)

    def _wait_files(self, paths):
        started = time.time()
        next_note = started + 30.0
        pending = [Path(path) for path in paths]
        while True:
            pending = [path for path in pending if not path.exists()]
            if not pending:
                return
            if self.timeout_s is not None and time.time() - started > self.timeout_s:
                raise RuntimeError(
                    "timed out waiting for {} farm file(s), including {}".format(
                        len(pending), pending[0].name
                    )
                )
            if self.is_root and time.time() >= next_note:
                print("  waiting for {} node(s)".format(len(pending)), flush=True)
                next_note = time.time() + 30.0
            time.sleep(self.poll_s)

    def _merge(self, step, n):
        merged = [None] * int(n)
        for rank in range(self.size):
            data = _load_npz(self.work / "shard_{}_{}.npz".format(step, rank))
            for local, index in enumerate(data["index"]):
                index = int(index)
                merged[index] = {
                    "ok": bool(data["ok"][local]),
                    "quanta": data["quanta"][local],
                    "quanta_ideal": data["quanta_ideal"][local],
                    "length_m": float(data["length_m"][local]),
                    "duration_s": float(data["duration_s"][local]),
                    "f_trans_MHz": (
                        float(data["frequency"][local, 0]),
                        float(data["frequency"][local, 1]),
                    ),
                    "message": str(data["message"][local]),
                }
        if any(item is None for item in merged):
            raise RuntimeError("a node returned no result for step {}".format(step))
        return merged
