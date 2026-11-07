"""Path-frame ion motion with voltage sets held between updates.

The well minimum sits on the design path. Its tangent is the axial direction
at ``F_AXIAL``. The normal and the vertical share the transverse curvature
left after the axial well is removed from the RF Laplacian. A voltage update
holds the minimum and the curvature from the last update. Optional noise is a
smooth electric field in the same frame, applied at every integration step.
Quanta are the secular energy about the continuous design path.
"""

import numpy as np

from .constants import CHARGE, HBAR, MASS, SIM_DT


def hold_steps(voltage_update_s, sim_dt=SIM_DT):
    """Integration steps that one target-DC update is held."""
    if voltage_update_s <= 0.0:
        raise ValueError("voltage update interval must be positive")
    if sim_dt <= 0.0:
        raise ValueError("simulation step must be positive")
    return max(int(round(voltage_update_s / sim_dt)), 1)


def path_directions(points):
    """Orthonormal frame, shape ``(n, 3, 3)``.

    Rows are tangent, in-plane normal, and vertical. For a path with no
    vertical slope this is the developed horizontal frame: tangent in the
    xy plane, normal ``(-ty, tx, 0)``, and vertical ``(0, 0, 1)``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("path must have shape (n, 3) with n >= 2")
    deriv = np.gradient(pts, axis=0)
    norm = np.linalg.norm(deriv, axis=1)
    missing = np.flatnonzero(norm == 0.0)
    if len(missing):
        good = np.flatnonzero(norm > 0.0)
        if good.size == 0:
            raise ValueError("path has no motion")
        nxt = np.searchsorted(good, missing).clip(0, len(good) - 1)
        deriv = deriv.copy()
        deriv[missing] = deriv[good[nxt]]
        norm = np.linalg.norm(deriv, axis=1)
    tangent = deriv / norm[:, None]

    up = np.array([0.0, 0.0, 1.0])
    vertical = up - np.sum(tangent * up, axis=1, keepdims=True) * tangent
    vnorm = np.linalg.norm(vertical, axis=1)
    bad = vnorm < 1e-8
    if np.any(bad):
        alt = np.array([1.0, 0.0, 0.0])
        vertical[bad] = alt - np.sum(tangent[bad] * alt, axis=1, keepdims=True) * tangent[bad]
        vnorm = np.linalg.norm(vertical, axis=1)
    vertical = vertical / vnorm[:, None]
    normal = np.cross(vertical, tangent)
    for i in range(1, len(pts)):
        if np.dot(normal[i], normal[i - 1]) < 0.0:
            normal[i] *= -1.0
    vertical = np.cross(tangent, normal)
    return np.stack([tangent, normal, vertical], axis=1)


def path_frame_noise(n_times, amplitude_Vm, seed):
    """Smooth field in the path frame, shape ``(n_times, 3)``.

    Columns are tangent, in-plane normal, and vertical. Each column is a slow
    random Fourier series, at most two cycles over the trajectory, multiplied
    by ``sin(pi tau)**2``. That window and its slope are zero at both ends.
    The series is then scaled so its maximum equals one draw from a Gaussian
    of standard deviation ``amplitude_Vm``.
    """
    n_times = int(n_times)
    if n_times < 2:
        raise ValueError("noise needs at least two samples")
    amplitude_Vm = float(amplitude_Vm)
    if amplitude_Vm < 0.0:
        raise ValueError("noise amplitude must be non-negative")
    if amplitude_Vm == 0.0:
        return np.zeros((n_times, 3))
    rng = np.random.default_rng(int(seed))
    tau = np.linspace(0.0, 1.0, n_times)
    coeffs = rng.normal(size=(5, 3))
    carrier = coeffs[0] + coeffs[1] * np.cos(2.0 * np.pi * tau)[:, None]
    carrier = carrier + coeffs[2] * np.sin(2.0 * np.pi * tau)[:, None]
    carrier = carrier + coeffs[3] * np.cos(4.0 * np.pi * tau)[:, None]
    carrier = carrier + coeffs[4] * np.sin(4.0 * np.pi * tau)[:, None]
    window = np.sin(np.pi * tau) ** 2
    series = carrier * window[:, None]
    peak = np.max(np.abs(series), axis=0, keepdims=True)
    series = series / np.maximum(peak, 1e-30)
    amplitude = rng.normal(scale=amplitude_Vm, size=(1, 3))
    return series * amplitude


def _force(position, center, direction, omega2_row):
    delta = position - center
    proj = direction @ delta
    return -(direction.T @ (omega2_row * proj))


def _noise_accel(directions, e_path):
    """Lab acceleration from a path-frame field, in m/s^2."""
    if e_path is None:
        return None
    e_path = np.asarray(e_path, dtype=float)
    if e_path.shape != (len(directions), 3):
        raise ValueError(
            "path-frame noise must have shape ({}, 3), got {}".format(
                len(directions), e_path.shape
            )
        )
    lab = np.einsum("nka,nk->na", directions, e_path)
    return (CHARGE / MASS) * lab


def _quanta(pos, vel, center, v_path, direction, omega2_row):
    delta = pos - center
    rel = vel - v_path
    disp = direction @ delta
    speed = direction @ rel
    energy = 0.5 * MASS * (speed ** 2 + omega2_row * disp ** 2)
    return energy / (HBAR * np.sqrt(omega2_row))


def integrate(r0, omega2, directions, n_hold, sim_dt=SIM_DT, e_path=None, r_init=None, v_init=None):
    """Velocity Verlet. ``n_hold`` 1 follows the path at every step.

    The well used for the force is constant for ``n_hold`` steps. Quanta use
    the continuous path. ``e_path`` is an electric field in V/m in the path
    frame at every sample, including while a voltage set is held.
    """
    r0 = np.asarray(r0, dtype=float)
    omega2 = np.asarray(omega2, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n_steps = len(r0) - 1
    if n_steps < 1:
        raise ValueError("path needs at least two samples")
    n_hold = max(int(n_hold), 1)
    v_raw = np.gradient(r0, sim_dt, axis=0)
    v_path = v_raw.copy()
    v_path[0] = 0.0
    v_path[-1] = 0.0
    a_noise = _noise_accel(directions, e_path)

    pos = r0[0].copy() if r_init is None else np.asarray(r_init, dtype=float).copy()
    vel = np.zeros(3) if v_init is None else np.asarray(v_init, dtype=float).copy()
    acc = _force(pos, r0[0], directions[0], omega2[0])
    if a_noise is not None:
        acc = acc + a_noise[0]

    stride = max(n_steps // 2000, 1)
    peak = np.zeros(3)
    final = np.zeros(3)
    for i in range(n_steps):
        vel_half = vel + 0.5 * sim_dt * acc
        pos = pos + sim_dt * vel_half
        step = i + 1
        j_force = min((step // n_hold) * n_hold, n_steps)
        acc = _force(pos, r0[j_force], directions[j_force], omega2[j_force])
        if a_noise is not None:
            acc = acc + a_noise[step]
        vel = vel_half + 0.5 * sim_dt * acc
        if step % stride == 0 or step == n_steps:
            q = _quanta(pos, vel, r0[step], v_path[step], directions[step], omega2[step])
            peak = np.maximum(peak, q)
            if step == n_steps:
                final = q
    return {
        "final_quanta": final,
        "peak_quanta": peak,
        "final_offset_um": (pos - r0[-1]) * 1e6,
    }


def integrate_acceleration(
    r0, omega2, directions, n_hold, acceleration, sim_dt=SIM_DT, e_path=None, r_init=None, v_init=None,
):
    """Velocity Verlet driven by ``acceleration(position_m, hold_index)``.

    ``hold_index`` is the path sample whose voltage set is held, the same
    schedule as ``integrate``. Quanta are still measured about the continuous
    design path. ``e_path`` is an extra field in V/m in that frame.
    """
    r0 = np.asarray(r0, dtype=float)
    omega2 = np.asarray(omega2, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n_steps = len(r0) - 1
    if n_steps < 1:
        raise ValueError("path needs at least two samples")
    n_hold = max(int(n_hold), 1)
    v_raw = np.gradient(r0, sim_dt, axis=0)
    v_path = v_raw.copy()
    v_path[0] = 0.0
    v_path[-1] = 0.0
    a_noise = _noise_accel(directions, e_path)

    def acc_at(pos, step):
        hold = min((step // n_hold) * n_hold, n_steps)
        acc = np.asarray(acceleration(pos, hold), dtype=float).reshape(3)
        if a_noise is not None:
            acc = acc + a_noise[step]
        return acc

    pos = r0[0].copy() if r_init is None else np.asarray(r_init, dtype=float).copy()
    vel = np.zeros(3) if v_init is None else np.asarray(v_init, dtype=float).copy()
    acc = acc_at(pos, 0)
    stride = max(n_steps // 2000, 1)
    peak = np.zeros(3)
    final = np.zeros(3)
    for i in range(n_steps):
        vel_half = vel + 0.5 * sim_dt * acc
        pos = pos + sim_dt * vel_half
        step = i + 1
        acc = acc_at(pos, step)
        vel = vel_half + 0.5 * sim_dt * acc
        if step % stride == 0 or step == n_steps:
            q = _quanta(pos, vel, r0[step], v_path[step], directions[step], omega2[step])
            peak = np.maximum(peak, q)
            if step == n_steps:
                final = q
    return {
        "final_quanta": final,
        "peak_quanta": peak,
        "final_offset_um": (pos - r0[-1]) * 1e6,
    }
