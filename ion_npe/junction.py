"""Y-junction electrode geometry and the RF pseudopotential on it.

The eight controls are the symmetric spline used by the developed junction
optimizer: ``ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l``.
"""

import os
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, CubicSpline

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplcache"),
)

try:
    from electrode import PolygonPixelElectrode, System
    from electrode.utils import shaped
except ImportError as err:
    raise ImportError(
        "The electrode package is required to evaluate the RF pseudopotential. "
        "On this machine it is installed in the trapsim conda environment."
    ) from err

from .constants import (
    CHARGE,
    KAPPA_AX,
    KAPPA_RAD,
    L_SCALE,
    MASS,
    OMEGA_AX,
    RF_SCALE,
    RF_SEP,
    RF_WIDTH,
    V_RF,
)


def generate_knot_vector(num_ctrlpts, degree=3):
    """Open uniform knot vector, matching the developed spline code."""
    n = num_ctrlpts - 1
    knots = [0.0] * (degree + 1)
    middle = [i / (n - degree + 1) for i in range(1, n - degree + 1)]
    knots.extend(middle)
    knots.extend([1.0] * (degree + 1))
    return np.asarray(knots, dtype=float)


def _rotate(points, angle):
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return points @ rotation.T


def _spline_curve(control_points, sample_size, degree=3):
    knots = generate_knot_vector(len(control_points), degree)
    spline_x = BSpline(knots, control_points[:, 0], degree)
    spline_y = BSpline(knots, control_points[:, 1], degree)
    u = np.linspace(0.0, 1.0, sample_size)
    return np.column_stack([spline_x(u), spline_y(u)])


def build_junction(control_points, rf_sep=RF_SEP, rf_w=RF_WIDTH, sample_size=100):
    """Electrode system for one Y junction. The RF electrode has ``rf = 1``."""
    controls = np.asarray(control_points, dtype=float).reshape(-1)
    if controls.size != 8:
        raise ValueError(
            "junction control points must be the 8 values "
            "(ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l)"
        )
    ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l = controls
    ao = ai + da
    angle = np.pi / 3.0
    point_co_top = np.array([
        np.cos(angle) * co_l + np.sin(angle) * (rf_sep / 2.0 + rf_w),
        np.sin(angle) * co_l - np.cos(angle) * (rf_sep / 2.0 + rf_w),
    ])
    point_ci_top = np.array([
        np.cos(angle) * ci_l + np.sin(angle) * (rf_sep / 2.0),
        np.sin(angle) * ci_l - np.cos(angle) * (rf_sep / 2.0),
    ])
    points_o = np.vstack([
        point_co_top,
        [bo_x, bo_y],
        [ao, 0.0],
        [bo_x, -bo_y],
        point_co_top * np.array([1.0, -1.0]),
    ])
    points_i = np.vstack([
        point_ci_top,
        [bi_x, bi_y],
        [ai, 0.0],
        [bi_x, -bi_y],
        point_ci_top * np.array([1.0, -1.0]),
    ])
    outer = _spline_curve(points_o, sample_size)
    inner = _spline_curve(points_i, sample_size)

    # Same chip extent as the developed builder: 10 * 2 * 2 * sqrt(3).
    lmax = 10 * 2 * 2 * np.sqrt(3.0)
    dist_to_edge = lmax / np.sqrt(3.0) / 2.0
    triangle_endpoint = np.array([[lmax / np.sqrt(3.0), 0.0]])
    outer_top = np.array([[
        np.cos(angle) * dist_to_edge + np.sin(angle) * (rf_sep / 2.0 + rf_w),
        np.sin(angle) * dist_to_edge - np.cos(angle) * (rf_sep / 2.0 + rf_w),
    ]])
    inner_top = np.array([[
        np.cos(angle) * dist_to_edge + np.sin(angle) * (rf_sep / 2.0),
        np.sin(angle) * dist_to_edge - np.cos(angle) * (rf_sep / 2.0),
    ]])
    outer_bot = outer_top * np.array([[1.0, -1.0]])
    inner_bot = inner_top * np.array([[1.0, -1.0]])
    rail_outer = np.concatenate([outer_top, outer, outer_bot])
    rail_inner = np.concatenate([inner_top, inner, inner_bot])
    rail = np.concatenate([rail_outer, rail_inner[::-1]])
    side = np.concatenate([triangle_endpoint, rail_outer, triangle_endpoint])

    rails, sides, center_bounds = [], [], []
    for turn in range(3):
        rotation = turn * 2.0 * np.pi / 3.0
        rails.append(_rotate(rail, rotation))
        sides.append(_rotate(side, rotation))
        center_bounds.append(_rotate(rail_inner, rotation)[::-1])
    center = np.concatenate(center_bounds)
    system = System([
        PolygonPixelElectrode(name="r", paths=[np.asarray(path, dtype=float) for path in rails]),
        PolygonPixelElectrode(name="dc_side", paths=[np.asarray(path, dtype=float) for path in sides]),
        PolygonPixelElectrode(name="dc_center", paths=[np.asarray(center, dtype=float)]),
    ])
    system["r"].rf = 1.0
    return system


def _stack_points(xs, ys, zs):
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    zs = np.asarray(zs, dtype=float).reshape(-1)
    if not xs.size == ys.size == zs.size:
        raise ValueError("x, y, z samples must have the same length")
    xyz = np.zeros((3, xs.size, 1, 1))
    xyz[0, :, 0, 0] = xs
    xyz[1, :, 0, 0] = ys
    xyz[2, :, 0, 0] = zs
    return xyz, xs.size


def laplacian_at(junction, xs, ys, zs):
    """Laplacian of the ``rf = 1`` pseudopotential. Inputs are 1-d and equal length."""
    xyz, n = _stack_points(xs, ys, zs)
    curv = np.asarray(shaped(junction.potential)(xyz, derivative=2))
    lap = np.trace(curv, axis1=-1, axis2=-2)
    return np.reshape(lap, (n,))


def path_frequencies(junction, xyz, tau, n_geom=120):
    """Squared secular frequencies in the path frame, and ``f_trans`` in Hz.

    Columns of ``omega2`` are tangent, normal, and vertical. The electrode
    Laplacian is scaled so the first sample has the target axial curvature
    plus two target radial curvatures. The axial part is assigned to the
    tangent. The remainder is split equally between the two transverse axes,
    so both start at ``F_RADIAL`` and then follow the junction.
    """
    xyz = np.asarray(xyz, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n = len(tau)
    n_geom = min(int(n_geom), n)
    geom_idx = np.unique(np.linspace(0, n - 1, n_geom).astype(int))
    if geom_idx.size < 4:
        geom_idx = np.arange(n)
    lap_geom = laplacian_at(
        junction, xyz[geom_idx, 0], xyz[geom_idx, 1], xyz[geom_idx, 2]
    )
    if not np.isfinite(lap_geom).all():
        raise ValueError("Laplacian is not finite along the path")
    lap_dim = CubicSpline(tau[geom_idx], lap_geom)(tau)
    lap_electrode = lap_dim * (V_RF * RF_SCALE) ** 2
    if lap_electrode[0] <= 0.0:
        raise ValueError(
            "Laplacian at the start of the path is not confining ({:.4g}).".format(
                float(lap_electrode[0])
            )
        )
    lap = lap_electrode * (KAPPA_AX + 2.0 * KAPPA_RAD) / lap_electrode[0]
    kappa_trans = 0.5 * (lap - KAPPA_AX)
    if np.any(kappa_trans <= 0.0):
        raise ValueError(
            "Transverse curvature is not confining. Minimum scaled Laplacian "
            "{:.4g}, axial curvature {:.4g}.".format(float(np.min(lap)), KAPPA_AX)
        )
    omega_trans = np.sqrt(CHARGE * kappa_trans / (MASS * L_SCALE ** 2))
    omega2 = np.column_stack([
        np.full(n, OMEGA_AX ** 2),
        omega_trans ** 2,
        omega_trans ** 2,
    ])
    return omega2, omega_trans / (2.0 * np.pi)
