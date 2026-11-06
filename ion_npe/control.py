"""DC voltages for a target well, and the field those voltages produce.

Each DC electrode is solved at unit voltage. Least squares then picks the
voltage set, held for one update interval, whose potential, gradient, and
Hessian match the transport target. The target gradient is zero at the design
point, so the minimum stays on the path. The Hessian is the secular curvature
in the path frame. The potential is the value produced by the gradient and
Hessian fit at the first update, and that same value is required at every
later update. Its row is scaled up to the field block, because a volt is
otherwise lost beside the curvature rows. The RF
pseudopotential is kept and subtracted, so the voltages supply only the
remainder. A penalty in the null space of those constraints drives the
~0.01 V spread on unused electrodes below a millivolt. Stray field is
not part of this fit.
"""

from dataclasses import dataclass

import numpy as np
from electrode.utils import expand_tensor

from .constants import CHARGE, KAPPA_AX, KAPPA_RAD, L_SCALE, MASS
from .junction import dc_electrode_names, laplacian_at


@dataclass
class TransportVoltages:
    """Voltages at each hold, in the order of ``names``."""

    voltages: np.ndarray
    update_indices: np.ndarray
    phi_target: float
    scale: float
    names: tuple
    residual: np.ndarray
    potential_error: np.ndarray


def _upper(hessian):
    """Upper triangle ``Hxx, Hxy, Hxz, Hyy, Hyz, Hzz`` of one 3×3 matrix."""
    return np.array([
        hessian[0, 0],
        hessian[0, 1],
        hessian[0, 2],
        hessian[1, 1],
        hessian[1, 2],
        hessian[2, 2],
    ], dtype=float)


def _rf_scale(junction, origin):
    """Scale that turns the ``rf = 1`` pseudopotential into the secular curvature.

    ``origin`` is one point in trap heights. The same scale is used everywhere,
    fixed by the Laplacian at the start of the path.
    """
    lap = float(laplacian_at(junction, origin[:1, 0], origin[:1, 1], origin[:1, 2])[0])
    if not np.isfinite(lap) or lap == 0.0:
        raise ValueError("RF Laplacian at the start of the path is not usable")
    return (KAPPA_AX + 2.0 * KAPPA_RAD) / lap


def _responses(junction, points):
    """Unit-volt DC responses and the scaled RF pseudopotential.

    ``points`` are trap heights, shape ``(n, 3)``. DC potential is in volts
    for a one-volt electrode. Gradients are per trap height. The RF arrays
    are already multiplied by the secular scale, so they are in the same units.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    names = dc_electrode_names()
    if tuple(junction.names[1:]) != names:
        raise ValueError("junction electrodes are not in DC name order")
    for electrode in junction:
        electrode.dc = 0.0
    phi = np.asarray(junction.individual_potential(points, 0), dtype=float)[1:, :, 0]
    grad = np.asarray(junction.individual_potential(points, 1), dtype=float)[1:]
    reduced = np.asarray(junction.individual_potential(points, 2), dtype=float)[1:]
    n_dc, n_points, _ = reduced.shape
    hessian = expand_tensor(reduced.reshape(-1, 5)).reshape(n_dc, n_points, 3, 3)
    rf_phi = np.asarray(junction.pseudo_potential(points, 0), dtype=float).reshape(n_points)
    rf_grad = np.asarray(junction.pseudo_potential(points, 1), dtype=float).reshape(n_points, 3)
    rf_hess = np.asarray(junction.pseudo_potential(points, 2), dtype=float).reshape(n_points, 3, 3)
    return phi, grad, hessian, rf_phi, rf_grad, rf_hess


# Voltages below this are where the L1 penalty linearizes. Entries that the
# minimum-norm fit had parked near 0.01 V fall below it.
_ARTIFACT_VOLT = 1e-3


def _fit(matrix, target):
    voltages, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if not np.isfinite(voltages).all():
        raise ValueError("voltage least squares did not return a finite solution")
    return voltages


def _penalized_fit(matrix, target):
    """Least squares, then an L1 penalty in the constraint null space.

    The minimum-norm solution spreads ~0.01 V onto electrodes that do not
    change the well. Moving the solution through the null space of the
    constraints removes that spread and leaves the potential, the gradient,
    and the Hessian where the first fit put them.
    """
    solved = _fit(matrix, target)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=True)
    if singular.size == 0 or singular[0] == 0.0:
        return solved
    rank = int(np.sum(singular > singular[0] * 1e-8))
    null = vt[rank:].T
    if null.shape[1] == 0:
        return solved
    for _ in range(12):
        weight = 1.0 / np.sqrt(np.abs(solved) + _ARTIFACT_VOLT)
        solved = solved + null @ _fit(weight[:, None] * null, -weight * solved)
    return solved


def solve_transport_voltages(junction, xyz, omega2, directions, n_hold):
    """Least-squares voltage set at each hold along ``xyz``.

    ``xyz`` is in trap heights. ``omega2`` and ``directions`` are the path
    frame used by the integrator. ``n_hold`` is the number of integration
    samples one voltage set covers.
    """
    xyz = np.asarray(xyz, dtype=float)
    omega2 = np.asarray(omega2, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n_hold = max(int(n_hold), 1)
    if len(xyz) < 2:
        raise ValueError("voltage solution needs at least two path samples")
    update_indices = np.arange(0, len(xyz), n_hold)
    points = xyz[update_indices]
    scale = _rf_scale(junction, xyz[:1])
    phi, grad, hessian, rf_phi, rf_grad, rf_hess = _responses(junction, points)
    kappa = omega2[update_indices] * MASS * L_SCALE ** 2 / CHARGE
    rf_phi = scale * rf_phi
    rf_grad = scale * rf_grad
    rf_hess = scale * rf_hess

    # SI rows: volts, V/m, V/m^2. This is the notebook's constraint order.
    grad_si = grad / L_SCALE
    hess_si = hessian / L_SCALE ** 2
    rf_grad_si = rf_grad / L_SCALE
    rf_hess_si = rf_hess / L_SCALE ** 2
    n_dc = phi.shape[0]
    n_updates = len(update_indices)
    matrix_field = np.empty((9, n_dc), dtype=float)
    matrix_full = np.empty((10, n_dc), dtype=float)
    voltages = np.empty((n_updates, n_dc), dtype=float)
    residual = np.empty(n_updates, dtype=float)
    potential_error = np.empty(n_updates, dtype=float)

    def field_target(index):
        frame = directions[update_indices[index]]
        hessian_lab = frame.T @ np.diag(kappa[index]) @ frame
        gradient = -rf_grad_si[index]
        curvature = _upper(hessian_lab / L_SCALE ** 2 - rf_hess_si[index])
        return gradient, curvature

    gradient0, curvature0 = field_target(0)
    for electrode in range(n_dc):
        matrix_field[:, electrode] = np.concatenate([grad_si[electrode, 0], _upper(hess_si[electrode, 0])])
    open_voltage = _penalized_fit(matrix_field, np.concatenate([gradient0, curvature0]))
    phi_target = float(rf_phi[0] + phi[:, 0] @ open_voltage)

    for index in range(n_updates):
        gradient, curvature = field_target(index)
        for electrode in range(n_dc):
            matrix_full[:, electrode] = np.concatenate([
                [phi[electrode, index]],
                grad_si[electrode, index],
                _upper(hess_si[electrode, index]),
            ])
        target = np.concatenate([
            [phi_target - rf_phi[index]],
            gradient,
            curvature,
        ])
        # A volt is tiny next to the V/m^2 curvature rows, so an unweighted
        # fit ignores the potential. Scale that one row up to the field block.
        # The curvature residual is unchanged: the extra voltage lies in the
        # null space of the gradient and the Hessian.
        field_norm = float(np.linalg.norm(matrix_full[1:]))
        potential_norm = float(np.linalg.norm(matrix_full[0]))
        weight = field_norm / max(potential_norm, 1e-15)
        matrix_full[0] *= weight
        target[0] *= weight
        solved = _penalized_fit(matrix_full, target)
        mismatch = matrix_full @ solved - target
        scale_b = max(float(np.linalg.norm(target)), 1.0)
        voltages[index] = solved
        residual[index] = float(np.linalg.norm(mismatch) / scale_b)
        potential_error[index] = float(rf_phi[index] + phi[:, index] @ solved - phi_target)
    if not np.isfinite(voltages).all():
        raise ValueError("transport voltages are not finite")
    return TransportVoltages(
        voltages=voltages,
        update_indices=update_indices,
        phi_target=phi_target,
        scale=float(scale),
        names=dc_electrode_names(),
        residual=residual,
        potential_error=potential_error,
    )


def electrode_acceleration(junction, voltages, scale, position_m):
    """Lab acceleration, m/s^2, from one held voltage set at ``position_m``.

    The DC field is the sum of the unit responses at the ion, each weighted
    by its voltage. The RF piece is the scaled pseudopotential at the same
    point. Height is whatever the ion has there.
    """
    point = np.asarray(position_m, dtype=float).reshape(1, 3) / L_SCALE
    dc = [electrode for electrode in junction if electrode.name != "r"]
    if len(dc) != len(voltages):
        raise ValueError("voltage vector does not match the DC electrodes")
    for electrode, voltage in zip(dc, voltages):
        electrode.dc = float(voltage)
    dc_grad = np.asarray(junction.electrical_potential(point, "dc", 1), dtype=float).reshape(3)
    rf_grad = np.asarray(junction.pseudo_potential(point, 1), dtype=float).reshape(3)
    gradient = dc_grad + float(scale) * rf_grad
    return -(CHARGE / (MASS * L_SCALE)) * gradient
