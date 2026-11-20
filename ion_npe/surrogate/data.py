"""Multi-field heating tables and design-grouped splits.

Each example is one joint design at one stray-field amplitude, with the
final quanta of every recorded noise realization. The amplitude is the value
stored in the archive. Designs that match across amplitudes, including the
nominal design shared by a field sweep, stay in one split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .layout import (
    JOINT_DIM,
    legal_bounds,
    quanta_to_log,
)


_AMPLITUDE_KEYS = ("noise_std_Vm", "noise_std", "field_std_Vm")


@dataclass
class FieldTable:
    """Design-field pairs. ``quanta`` is NaN where that realization is invalid."""

    design: np.ndarray
    sigma: np.ndarray
    quanta: np.ndarray
    valid: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    source: np.ndarray
    info: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.design.shape[0])

    def log_quanta(self, eps=None):
        return quanta_to_log(self.quanta) if eps is None else quanta_to_log(self.quanta, eps)


@dataclass
class SplitTable:
    table: FieldTable
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    interp: np.ndarray
    groups: np.ndarray
    holdout_fields: np.ndarray
    seed: int
    key_decimals: int
    train_mean: np.ndarray
    train_std: np.ndarray

    def fingerprint(self):
        return _fingerprint(self.table.design, self.table.sigma)

    def indices(self, name):
        if name not in ("train", "val", "test", "interp"):
            raise KeyError(name)
        return np.asarray(getattr(self, name), dtype=int)


def load_field_data(paths, amplitudes=None):
    """Load one or more archives into a single table.

    ``paths`` may be ``joint.npz`` files or directories that contain them.
    A stored ``noise_std_Vm`` is used as the amplitude. ``amplitudes``, when
    given, supplies a value only for an archive that does not already store
    one, and it must agree with a stored value when both exist.
    """
    files = _expand_paths(paths)
    if not files:
        raise FileNotFoundError("no joint archives found in {}".format(paths))
    overrides = _override_map(files, amplitudes)
    lower, upper = legal_bounds()
    designs = []
    sigmas = []
    quanta = []
    sources = []
    dropped = 0
    partial = 0
    seen_bounds = None
    for path in files:
        archive = np.load(path, allow_pickle=False)
        if "quanta" not in archive.files or ("rows" not in archive.files and "x" not in archive.files):
            continue
        rows = np.asarray(archive["rows"] if "rows" in archive.files else archive["x"], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != JOINT_DIM:
            raise ValueError(
                "{} has design shape {}. This surrogate expects the current {}-parameter "
                "joint vector (8 junction, cross_s, 7 heights, 4 gates).".format(
                    path, rows.shape, JOINT_DIM
                )
            )
        heating = _as_samples(archive["quanta"], len(rows), path)
        sigma = _archive_sigma(archive, path, overrides.get(str(path)))
        file_bounds = _archive_bounds(archive, path, rows.shape[1])
        if file_bounds is not None:
            if seen_bounds is None:
                seen_bounds = file_bounds
            elif not (
                np.allclose(seen_bounds[0], file_bounds[0]) and np.allclose(seen_bounds[1], file_bounds[1])
            ):
                raise ValueError("archives disagree on the joint bounds: {}".format(path))
        sigma_col = np.full(len(rows), float(sigma), dtype=np.float64)
        if sigma_col.shape[0] != len(rows):
            raise ValueError("amplitude and designs disagree in {}".format(path))
        valid = np.isfinite(heating).all(axis=-1) & (heating >= 0.0).all(axis=-1)
        keep = valid.any(axis=1)
        dropped += int((~keep).sum())
        partial += int((keep & ~valid.all(axis=1)).sum())
        if not np.any(keep):
            continue
        designs.append(rows[keep])
        sigmas.append(sigma_col[keep])
        kept_q = heating[keep]
        kept_v = valid[keep]
        kept_q = kept_q.copy()
        kept_q[~kept_v] = np.nan
        quanta.append(kept_q)
        sources.append(np.full(int(keep.sum()), str(path), dtype=object))
    if not designs:
        raise ValueError("every design-field pair was invalid; nothing was imputed")
    if seen_bounds is not None:
        lower, upper = seen_bounds
    design = np.vstack(designs)
    sigma = np.concatenate(sigmas)
    stacked, valid = _pad_samples(quanta)
    info = {
        "n_dropped_invalid": int(dropped),
        "n_partial": int(partial),
        "files": [str(path) for path in files if _is_heating_archive(path)],
        "amplitudes": [float(v) for v in np.unique(np.round(sigma, 8))],
        "bounds_source": "archive" if seen_bounds is not None else "design_space",
    }
    return FieldTable(
        design=design,
        sigma=sigma,
        quanta=stacked,
        valid=valid,
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        source=np.concatenate(sources),
        info=info,
    )


def make_split(
    table,
    holdout_fields=(),
    seed=0,
    fractions=(0.70, 0.15, 0.15),
    key_decimals=5,
):
    """Group by design, then hold out intermediate field amplitudes.

    Rows whose amplitude is in ``holdout_fields`` are removed before the
    design split and returned as ``interp``. A design that also appears at a
    trained amplitude can still land in train: that is the interpolation set.
    The same design never appears in both train and test.
    """
    if len(table) < 3:
        raise ValueError("need at least three design-field pairs to split")
    train_frac, val_frac, test_frac = (float(v) for v in fractions)
    if min(train_frac, val_frac, test_frac) <= 0.0 or abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("split fractions must be positive and sum to 1")
    holdout = np.asarray(list(holdout_fields), dtype=np.float64).reshape(-1)
    held = _held_mask(table.sigma, holdout)
    groups = design_groups(table.design, table.lower, table.upper, key_decimals)
    eligible = np.flatnonzero(~held)
    if eligible.size < 3:
        raise ValueError("holdout leaves fewer than three design-field pairs")
    eligible_groups = np.unique(groups[eligible])
    order = np.random.default_rng(int(seed)).permutation(eligible_groups.size)
    n_test = max(int(round(test_frac * eligible_groups.size)), 1)
    n_val = max(int(round(val_frac * eligible_groups.size)), 1)
    if n_test + n_val >= eligible_groups.size:
        n_test = 1
        n_val = 1
    test_groups = set(eligible_groups[order[:n_test]].tolist())
    val_groups = set(eligible_groups[order[n_test:n_test + n_val]].tolist())
    train_groups = set(eligible_groups[order[n_test + n_val:]].tolist())
    train = np.flatnonzero(~held & np.isin(groups, list(train_groups)))
    val = np.flatnonzero(~held & np.isin(groups, list(val_groups)))
    test = np.flatnonzero(~held & np.isin(groups, list(test_groups)))
    interp = np.flatnonzero(held)
    mean = table.design[train].mean(axis=0)
    std = table.design[train].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return SplitTable(
        table=table,
        train=train.astype(int),
        val=val.astype(int),
        test=test.astype(int),
        interp=interp.astype(int),
        groups=groups.astype(int),
        holdout_fields=holdout,
        seed=int(seed),
        key_decimals=int(key_decimals),
        train_mean=mean,
        train_std=std,
    )


def design_groups(design, lower, upper, decimals=5):
    """Integer id shared by identical or nearly identical design vectors."""
    design = np.asarray(design, dtype=np.float64)
    span = np.maximum(np.asarray(upper) - np.asarray(lower), 1e-8)
    unit = np.round((design - np.asarray(lower)) / span, int(decimals))
    _unique, inverse = np.unique(unit, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def field_sample_weights(sigma):
    """Inverse-frequency weight so each stray-field amplitude is sampled equally."""
    labels = np.round(np.asarray(sigma, dtype=np.float64), 8)
    _unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    return (1.0 / counts[inverse]).astype(np.float64)


def table_fingerprint(table):
    return _fingerprint(table.design, table.sigma)


def describe(table):
    labels, counts = np.unique(np.round(table.sigma, 8), return_counts=True)
    parts = ["{} designs".format(len(table))]
    for label, count in zip(labels, counts):
        parts.append("{:.4g} V/m x {}".format(float(label), int(count)))
    dropped = table.info.get("n_dropped_invalid", 0)
    if dropped:
        parts.append("{} invalid pairs excluded".format(int(dropped)))
    return ", ".join(parts)


def _expand_paths(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            found.extend(sorted(item for item in path.rglob("*.npz") if _is_heating_archive(item)))
        elif _is_heating_archive(path):
            found.append(path)
    # Stable, unique.
    unique = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _is_heating_archive(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            files = set(archive.files)
    except Exception:
        return False
    return "quanta" in files and ("rows" in files or "x" in files)


def _override_map(files, amplitudes):
    if amplitudes is None:
        return {}
    values = [float(v) for v in amplitudes]
    heating = [path for path in files if _is_heating_archive(path)]
    if len(values) == 1 and len(heating) > 1:
        return {str(path): values[0] for path in heating}
    if len(values) != len(heating):
        raise ValueError(
            "got {} amplitudes for {} archives; pass one per archive or a single value".format(
                len(values), len(heating)
            )
        )
    return {str(path): value for path, value in zip(heating, values)}


def _archive_sigma(archive, path, override):
    stored = None
    for key in _AMPLITUDE_KEYS:
        if key in archive.files:
            stored = _as_sigma(archive[key], path, key)
            break
    if stored is None:
        summary = _summary_sigma(Path(path).parent / "summary.txt")
        if summary is not None:
            stored = float(summary)
            source = "summary.txt"
        else:
            source = None
    else:
        source = "archive"
    if stored is None and override is None:
        raise ValueError(
            "{} does not store noise_std_Vm. Pass an amplitude explicitly; "
            "it will not be read out of a design coordinate.".format(path)
        )
    if stored is not None and override is not None and abs(float(stored) - float(override)) > 1e-8:
        raise ValueError(
            "{} stores {:.6g} V/m, which disagrees with the requested {:.6g}".format(
                path, float(stored), float(override)
            )
        )
    if stored is None:
        return float(override)
    return float(stored)


def _as_sigma(value, path, key):
    array = np.asarray(value, dtype=np.float64)
    if array.shape == () or array.size == 1:
        sigma = float(array.reshape(-1)[0])
    else:
        raise ValueError(
            "{} field {} has shape {}; expected one amplitude for the archive".format(
                path, key, array.shape
            )
        )
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("{} has an invalid stray-field amplitude".format(path))
    return sigma


def _summary_sigma(path):
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        text = line.strip().lower()
        if not text.startswith("stray field:"):
            continue
        body = text.split(":", 1)[1].strip()
        if body.startswith("off"):
            return 0.0
        token = body.split()[0]
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _archive_bounds(archive, path, dim):
    if "lower" in archive.files and "upper" in archive.files:
        lower = np.asarray(archive["lower"], dtype=np.float64).reshape(-1)
        upper = np.asarray(archive["upper"], dtype=np.float64).reshape(-1)
        if lower.size == dim and upper.size == dim:
            return lower, upper
    parent = Path(path).parent.parent / "designs.npz"
    sibling = Path(path).with_name("designs.npz")
    for candidate in (sibling, parent):
        if not candidate.exists():
            continue
        with np.load(candidate, allow_pickle=False) as extra:
            if "lower" not in extra.files or "upper" not in extra.files:
                continue
            lower = np.asarray(extra["lower"], dtype=np.float64).reshape(-1)
            upper = np.asarray(extra["upper"], dtype=np.float64).reshape(-1)
            if lower.size == dim and upper.size == dim:
                return lower, upper
    return None


def _as_samples(quanta, n_rows, path):
    array = np.asarray(quanta, dtype=np.float64)
    if array.ndim == 2 and array.shape == (n_rows, 3):
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[0] != n_rows or array.shape[-1] != 3:
        raise ValueError("{} quanta has shape {}, expected (N, seeds, 3)".format(path, array.shape))
    return array


def _pad_samples(blocks):
    width = max(block.shape[1] for block in blocks)
    rows = sum(block.shape[0] for block in blocks)
    stacked = np.full((rows, width, 3), np.nan, dtype=np.float64)
    valid = np.zeros((rows, width), dtype=bool)
    cursor = 0
    for block in blocks:
        n, seeds, _modes = block.shape
        finite = np.isfinite(block).all(axis=-1)
        stacked[cursor:cursor + n, :seeds] = block
        valid[cursor:cursor + n, :seeds] = finite
        cursor += n
    return stacked, valid


def _held_mask(sigma, holdout):
    if holdout.size == 0:
        return np.zeros(len(sigma), dtype=bool)
    distance = np.min(np.abs(sigma[:, None] - holdout[None, :]), axis=1)
    return distance <= 1e-8


def _fingerprint(design, sigma):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(design, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(sigma, dtype=np.float64).tobytes())
    return digest.hexdigest()


def split_payload(split):
    return {
        "fingerprint": split.fingerprint(),
        "train": split.train,
        "val": split.val,
        "test": split.test,
        "interp": split.interp,
        "groups": split.groups,
        "holdout_fields": split.holdout_fields,
        "seed": np.int64(split.seed),
        "key_decimals": np.int64(split.key_decimals),
        "train_mean": split.train_mean,
        "train_std": split.train_std,
        "lower": split.table.lower,
        "upper": split.table.upper,
        "info": json.dumps(split.table.info),
    }
