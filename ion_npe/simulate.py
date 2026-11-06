"""Integrate one junction and one three-dimensional trajectory.

The ion moves in the RF pseudopotential of the junction and in a target DC
well along the design path. The DC well is not taken from a solved electrode
voltage set. It is the target harmonic well, held for 100 ns. The stray field
is off unless ``noise_std_Vm`` is set.
"""

from dataclasses import dataclass

import numpy as np

from .constants import (
    L_SCALE,
    NOISE_SEEDS,
    NOISE_STD_VM,
    SIM_DT,
    TRANSPORT_SPEED_M_S,
    VOLTAGE_UPDATE_S,
)
from .dynamics import hold_steps, integrate, path_directions, path_frame_noise
from .junction import build_junction, path_frequencies
from .trajectory import dynamics_grid


@dataclass
class RunResult:
    """Final heating for one design. Quanta are tangent, normal, vertical."""

    ok: bool
    quanta: np.ndarray
    quanta_ideal: np.ndarray
    peak_quanta: np.ndarray
    noise_max_Vm: np.ndarray
    message: str
    duration_s: float
    length_m: float
    f_trans_MHz: tuple
    z_range: tuple
    n_hold: int


def _fail(message):
    nan3 = np.full(3, np.nan)
    return RunResult(
        ok=False,
        quanta=nan3.reshape(1, 3),
        quanta_ideal=nan3.reshape(1, 3),
        peak_quanta=nan3.reshape(1, 3),
        noise_max_Vm=nan3.reshape(1, 3),
        message=message,
        duration_s=np.nan,
        length_m=np.nan,
        f_trans_MHz=(np.nan, np.nan),
        z_range=(np.nan, np.nan),
        n_hold=0,
    )


def simulation_settings(simulate_kwargs=None):
    """Noise amplitude and the samples drawn for one design.

    A positive stray field needs ``NOISE_SEEDS`` samples. One sample has no
    variance, so the stochastic network cannot be trained on it.
    """
    kwargs = dict(simulate_kwargs or {})
    kwargs.setdefault("noise_std_Vm", NOISE_STD_VM)
    amplitude = float(kwargs["noise_std_Vm"])
    if amplitude < 0.0:
        raise ValueError("noise amplitude must be non-negative")
    seeds = tuple(int(seed) for seed in kwargs.get("seeds", NOISE_SEEDS))
    if not seeds:
        raise ValueError("noise seeds are empty")
    if amplitude > 0.0 and len(seeds) < len(NOISE_SEEDS):
        raise ValueError(
            "stray field {:.4g} V/m needs at least {} samples per design "
            "for the stochastic network; got {}".format(amplitude, len(NOISE_SEEDS), len(seeds))
        )
    kwargs["seeds"] = seeds
    return kwargs


def stray_field_line(simulate_kwargs):
    """One summary line for the stray field actually used."""
    amplitude = float(simulate_kwargs["noise_std_Vm"])
    if amplitude == 0.0:
        return "stray field: off"
    return "stray field: {:.4g} V/m, {} samples per design".format(
        amplitude, len(simulate_kwargs["seeds"])
    )


def simulate_design(
    design,
    seeds=(0,),
    speed_m_s=TRANSPORT_SPEED_M_S,
    voltage_update_s=VOLTAGE_UPDATE_S,
    noise_std_Vm=NOISE_STD_VM,
    with_ideal=False,
    sim_dt=SIM_DT,
    n_geom=120,
):
    """Final motional quanta for ``design``.

    ``seeds`` each draw an independent stray field. With ``noise_std_Vm`` 0
    the field is omitted and one trajectory is integrated. ``with_ideal``
    also integrates the well that follows the path at every step.
    """
    try:
        sampled = dynamics_grid(design, speed_m_s, L_SCALE, sim_dt)
        junction = build_junction(design.junction)
        xyz = sampled["xyz"]
        omega2, f_trans = path_frequencies(junction, xyz, sampled["tau"], n_geom=n_geom)
        directions = path_directions(xyz)
        r0 = xyz * L_SCALE
        n_hold = hold_steps(voltage_update_s, sim_dt)
        amplitude = float(noise_std_Vm)
        if amplitude < 0.0:
            raise ValueError("noise amplitude must be non-negative")
        seed_list = [int(seed) for seed in seeds]
        if not seed_list:
            raise ValueError("noise seeds are empty")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as err:
        return _fail(str(err))

    quanta = []
    ideal_rows = []
    peaks = []
    noise_max = []
    try:
        quiet = None
        for seed in seed_list:
            e_path = None if amplitude == 0.0 else path_frame_noise(len(r0), amplitude, seed)
            if amplitude == 0.0 and quiet is not None:
                physical = quiet
            else:
                physical = integrate(
                    r0, omega2, directions, n_hold=n_hold, sim_dt=sim_dt, e_path=e_path
                )
                if amplitude == 0.0:
                    quiet = physical
            quanta.append(np.array(physical["final_quanta"], copy=True))
            peaks.append(np.array(physical["peak_quanta"], copy=True))
            if e_path is None:
                noise_max.append(np.zeros(3))
            else:
                noise_max.append(np.max(np.abs(e_path), axis=0))
            if with_ideal:
                ideal = integrate(r0, omega2, directions, n_hold=1, sim_dt=sim_dt, e_path=e_path)
                ideal_rows.append(ideal["final_quanta"])
            if not np.isfinite(physical["final_quanta"]).all():
                raise ValueError("heating is not finite")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as err:
        return _fail(str(err))

    quanta = np.vstack(quanta)
    peaks = np.vstack(peaks)
    noise_max = np.vstack(noise_max)
    if with_ideal:
        quanta_ideal = np.vstack(ideal_rows)
    else:
        quanta_ideal = np.full_like(quanta, np.nan)
    return RunResult(
        ok=True,
        quanta=quanta,
        quanta_ideal=quanta_ideal,
        peak_quanta=peaks,
        noise_max_Vm=noise_max,
        message="",
        duration_s=float(sampled["duration_s"]),
        length_m=float(sampled["length_m"]),
        f_trans_MHz=(float(f_trans.min() / 1e6), float(f_trans.max() / 1e6)),
        z_range=(float(xyz[:, 2].min()), float(xyz[:, 2].max())),
        n_hold=n_hold,
    )


def total_heating(quanta):
    """Mean over noise seeds of the sum of the three final quanta."""
    summed = np.sum(np.asarray(quanta, dtype=float), axis=-1)
    return np.mean(summed, axis=-1)
