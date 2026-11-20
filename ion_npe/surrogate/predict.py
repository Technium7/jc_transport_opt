"""Prediction interface for CEM screening and active learning.

Low predicted heating is not evidence that a candidate is physically valid.
Admissibility is checked with the same geometric and waveform constraints as
the joint search. Simulation remains the authority.
"""

from __future__ import annotations

import numpy as np

from ion_npe.design import (
    DesignSpace,
    N_FREE_GATES,
    N_TIME_GATES,
    expand_free_gates,
    within_transport_caps,
)
from ion_npe.trajectory import controls_stay_ordered, height_within_cap, s_of_tau, time_gate_samples

from .layout import GATE_LIMIT, JOINT_DIM


def joint_is_admissible(row, limit=GATE_LIMIT):
    """Geometry caps, a non-folding control polygon, and a monotone waveform."""
    values = np.asarray(row, dtype=float).reshape(-1)
    space = DesignSpace()
    if values.size != space.dim + N_FREE_GATES:
        raise ValueError("expected a {}-parameter joint vector".format(JOINT_DIM))
    design = space.unpack(values[:space.dim])
    design.gate_offsets = expand_free_gates(values[-N_FREE_GATES:])
    if not (
        within_transport_caps(design)
        and controls_stay_ordered(design)
        and height_within_cap(design)
    ):
        return False
    return _waveform_ok(design, limit)


def rank_candidates(
    predictor,
    rows,
    sigma,
    objective="expected",
    alpha=0.9,
    n_samples=256,
    rng=0,
    check_admissible=True,
):
    """Order candidates from best to worst for one stray-field amplitude.

    ``objective`` is ``expected``, ``quantile``, or ``cvar``. Inadmissible
    rows sort last. The returned scores are total heating in linear quanta.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    sigma = np.full(len(rows), float(sigma)) if np.ndim(sigma) == 0 else np.asarray(sigma, dtype=float)
    scores = _objective(predictor, rows, sigma, objective, alpha, n_samples, rng)
    admissible = np.ones(len(rows), dtype=bool)
    if check_admissible:
        admissible = np.array([joint_is_admissible(row) for row in rows], dtype=bool)
        scores = np.where(admissible, scores, np.inf)
    order = np.argsort(scores, kind="mergesort")
    return order, scores, admissible


def epistemic_acquisition(predictor, rows, sigma, phonon_quantile=0.5, check_admissible=True):
    """Active-learning order. Uncertainty is ensemble disagreement, not mixture variance."""
    from ion_npe.active import active_ranking

    rows = np.asarray(rows, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    epistemic = predictor.epistemic_total_std(rows, float(sigma))
    if epistemic is None:
        raise ValueError(
            "epistemic acquisition needs a deep ensemble; mixture variance is aleatoric heating, not model uncertainty"
        )
    mean = predictor.expected_total(rows, float(sigma))
    eligible = np.ones(len(rows), dtype=bool)
    if check_admissible:
        eligible = np.array([joint_is_admissible(row) for row in rows], dtype=bool)
    if not np.any(eligible):
        return np.array([], dtype=int)
    local = np.flatnonzero(eligible)
    order = active_ranking(mean[local], epistemic[local], phonon_quantile)
    return local[np.asarray(order, dtype=int)]


def _objective(predictor, rows, sigma, objective, alpha, n_samples, rng):
    if objective == "expected":
        return np.asarray(predictor.expected_total(rows, sigma), dtype=float)
    if objective == "quantile":
        quantiles = predictor.total_quantiles(rows, sigma, [alpha], n_samples=n_samples, rng=rng)
        return np.asarray(quantiles, dtype=float).reshape(len(rows), -1)[:, 0]
    if objective == "cvar":
        return np.asarray(predictor.cvar(rows, sigma, alpha=alpha, n_samples=n_samples, rng=rng), dtype=float)
    raise ValueError("objective must be expected, quantile, or cvar")


def _waveform_ok(design, limit, gap=0.015):
    offsets = np.asarray(design.gate_offsets, dtype=float).reshape(-1)
    n_gates = int(design.n_time_gates)
    if n_gates != int(N_TIME_GATES) or offsets.size != n_gates - 1:
        return False
    if np.any(np.abs(offsets) > float(limit) + 1e-12):
        return False
    _knots, samples = time_gate_samples(design.curves, n_gates, offsets)
    if np.any(np.diff(samples) < float(gap)):
        return False
    tau = np.linspace(0.0, 1.0, 241)
    distance = s_of_tau(tau, design.curves, n_gates, offsets)
    if distance[0] < -1e-6 or distance[-1] > 1.0 + 1e-6:
        return False
    return bool(np.all(np.diff(distance) >= -1e-8))
