"""Feasibility of a phase-space response surrogate.

The heating model in this package predicts motional quanta directly. A more
structured alternative is to predict the terminal phase-space deviation and
then apply the known mode-energy formula:

    u_f = u_0(d) + sigma_E R(d) phi + Delta u_nonlinear
    n = energy(u_f) / (hbar omega)

``phi`` is the stray-field realization. The integrator already builds that
realization as a five-term Fourier series on each path-frame axis, then
rescales it to a Gaussian amplitude. Those coefficients are not returned, and
the saved field sweeps do not store terminal velocity, mode frequency, or the
field coefficients. Training this response model requires a new collection.
This module does not change the integrator.
"""

from __future__ import annotations

import numpy as np

from ion_npe.dynamics import _quanta, integrate, path_frame_noise


REQUIRED_FROM_SIMULATOR = (
    "terminal relative position",
    "terminal relative velocity",
    "terminal mode frequencies or omega^2",
    "stray-field Fourier coefficients and per-axis amplitudes",
)


def quanta_from_state(pos, vel, center, v_path, direction, omega2_row):
    """Motional quanta from the existing mode-energy formula."""
    return _quanta(pos, vel, center, v_path, direction, omega2_row)


def audit_integrator():
    """Report which phase-space channels a tiny integration actually returns."""
    n = 5
    distance = np.linspace(0.0, 1e-5, n)
    path = np.column_stack([distance, np.zeros(n), np.full(n, 50e-6)])
    omega = 2.0 * np.pi * 1.0e6
    omega2 = np.full((n, 3), omega ** 2)
    directions = np.repeat(np.eye(3)[None, :, :], n, axis=0)
    result = integrate(path, omega2, directions, n_hold=1, sim_dt=1e-9)
    noise = path_frame_noise(n, 1.0, seed=0)
    present = set(result)
    return {
        "integrator_keys": sorted(present),
        "records_final_quanta": "final_quanta" in present,
        "records_final_offset": "final_offset_um" in present,
        "records_terminal_velocity": "final_velocity" in present or "terminal_velocity" in present,
        "records_mode_frequency": any("omega" in key or "freq" in key for key in present),
        "noise_is_a_field_series_only": tuple(noise.shape) == (n, 3),
        "records_field_coefficients": False,
        "missing": list(REQUIRED_FROM_SIMULATOR),
        "note": (
            "path_frame_noise draws 5 Fourier coefficients per axis and 3 amplitudes "
            "internally and returns only the sampled field. Existing npz archives store "
            "final quanta, not the terminal phase-space state or those coefficients."
        ),
    }
