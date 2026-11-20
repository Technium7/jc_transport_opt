"""Tokens and bounds for the current 20-parameter joint design.

Order, matching ``DesignSpace.pack`` plus the four outbound gates:

- 8 junction controls
- bend radius ``cross_s``
- 7 height controls ``h0_z`` ... ``h6_z``
- 4 waveform-gate offsets

``(a, b)`` is fixed at ``(3, 1.5)`` and is not a coordinate. Height stations
are equal steps of inbound arc length. ``h0_z`` is the first station after
the fixed 50 um start, and ``h6_z`` is the crossing. Gate positions are the
nominal normalized times of the four outbound spline gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ion_npe.design import DesignSpace, N_FREE_GATES, N_HEIGHT, N_TIME_GATES
from ion_npe.model import LOG_FLOOR
from ion_npe.trajectory import height_fractions

# Same limit as ion_npe.joint.GATE_LIMIT. Kept here so this module does not
# import the simulator or the electrode package.
GATE_LIMIT = 0.10
LOG_EPS = float(LOG_FLOOR)
E_REF_DEFAULT = 10.0

GROUP_JUNCTION = "junction"
GROUP_TRAJECTORY = "trajectory"
GROUP_WAVEFORM = "waveform"
GROUP_FIELD = "field"
GROUP_CLS = "cls"


@dataclass(frozen=True)
class TokenSpec:
    name: str
    group: str
    column: int
    position: float


def legal_bounds():
    """Physical box of one joint vector, shape ``(20,)`` each."""
    space = DesignSpace()
    lower = np.concatenate([space.lower, np.full(N_FREE_GATES, -GATE_LIMIT)])
    upper = np.concatenate([space.upper, np.full(N_FREE_GATES, GATE_LIMIT)])
    return lower.astype(np.float64), upper.astype(np.float64)


def joint_dim():
    return int(DesignSpace().dim + N_FREE_GATES)


JOINT_DIM = 20


def joint_names():
    space = DesignSpace()
    return tuple(space.names) + tuple("g{}".format(i) for i in range(N_FREE_GATES))


def height_positions():
    """Arc-length fraction of each free height. The start itself is not free."""
    stations = np.asarray(height_fractions(None), dtype=np.float64)
    if stations.size != N_HEIGHT + 1:
        raise RuntimeError("height stations must be the start plus {} controls".format(N_HEIGHT))
    return stations[1:]


def gate_positions():
    """Normalized time of each stored outbound gate. The center gate is omitted."""
    knots = np.linspace(0.0, 1.0, int(N_TIME_GATES) + 1)
    return knots[1:1 + N_FREE_GATES].astype(np.float64)


def joint_tokens():
    """One token per joint coordinate, in vector order."""
    names = joint_names()
    if len(names) != joint_dim():
        raise RuntimeError("joint names and dimension disagree")
    heights = height_positions()
    gates = gate_positions()
    tokens = []
    n_height = 0
    n_gate = 0
    for column, name in enumerate(names):
        if column < 8:
            group = GROUP_JUNCTION
            position = 0.0
        elif name == "cross_s":
            group = GROUP_TRAJECTORY
            position = 1.0
        elif name.endswith("_z"):
            group = GROUP_TRAJECTORY
            position = float(heights[n_height])
            n_height += 1
        else:
            group = GROUP_WAVEFORM
            position = float(gates[n_gate])
            n_gate += 1
        tokens.append(TokenSpec(name, group, column, position))
    if n_height != N_HEIGHT or n_gate != N_FREE_GATES:
        raise RuntimeError("joint tokenization did not consume the height and gate blocks")
    return tuple(tokens)


def normalize_design(design, lower, upper):
    """Map each coordinate from its physical box onto ``[-1, 1]``."""
    design = np.asarray(design, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    span = np.maximum(upper - lower, 1e-8)
    return (2.0 * (design - lower) / span - 1.0).astype(np.float32)


def field_features(sigma, e_ref=E_REF_DEFAULT):
    """``[log(1 + sigma/E_ref), (sigma/E_ref)^2]``. Zero field is the zero vector.

    The square is a linear-response feature. The network is not constrained
    to increase heating with field strength.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if np.any(sigma < 0.0):
        raise ValueError("stray-field amplitude must be non-negative")
    e_ref = float(e_ref)
    if e_ref <= 0.0:
        raise ValueError("E_ref must be positive")
    ratio = sigma / e_ref
    feat = np.stack([np.log1p(ratio), ratio ** 2], axis=-1)
    return feat.astype(np.float32)


def quanta_to_log(quanta, eps=LOG_EPS):
    """``log10(n + eps)`` for finite, non-negative quanta. Invalid entries stay NaN."""
    quanta = np.asarray(quanta, dtype=np.float64)
    out = np.full(quanta.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(quanta) & (quanta >= 0.0)
    out[ok] = np.log10(quanta[ok] + float(eps))
    return out


def log_to_quanta(log_quanta, eps=LOG_EPS):
    """Inverse of :func:`quanta_to_log`. Negative results are clipped to zero."""
    raw = np.power(10.0, np.asarray(log_quanta, dtype=np.float64)) - float(eps)
    return np.maximum(raw, 0.0)
