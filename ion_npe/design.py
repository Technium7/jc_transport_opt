"""Junction and trajectory design vector.

The search coordinates are the eight symmetric Y-junction controls, the bend
radius ``cross_s``, the heights of the bend controls, and the ``(a, b)``
time profile. ``cross_s`` fixes the bend ``(x, y)``. Those coordinates are
not separate search parameters. ``cross_s`` stays within 15 um of the RF
center, and each height stays within 15 um of 50 um. The arm stays on the
developed path.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    CROSS_CAP_M,
    HEIGHT_CAP_M,
    ION_HEIGHT,
    L_SCALE,
)

# Fixed arm controls. They are not free parameters; the path still uses them.
_ARM = (
    (-4.50, 0.0, ION_HEIGHT),
    (-3.55, 0.0, ION_HEIGHT),
    (-2.60, 0.0, ION_HEIGHT),
    (-1.70, 0.0, ION_HEIGHT),
)
N_ARM = len(_ARM)

# Stations of the developed bend, used only to keep the same sample spacing.
# The coordinates themselves are rebuilt from the tangent-matched curve.
_DEVELOPED_STATIONS = np.array([
    [-0.7263, 0.0009],
    [-0.5925, 0.0034],
    [-0.4762, 0.0076],
    [-0.3751, 0.0133],
    [-0.2874, 0.0207],
    [-0.2114, 0.0302],
    [-0.1459, 0.0429],
], dtype=float)
N_INTERIOR = len(_DEVELOPED_STATIONS)
_Z_SPAN = HEIGHT_CAP_M / L_SCALE
_Z_LO = ION_HEIGHT - _Z_SPAN
_Z_HI = ION_HEIGHT + _Z_SPAN
_CROSS_S = 0.1
_CROSS_HI = CROSS_CAP_M / L_SCALE
_MIRROR_ANGLE = 2.0 * np.pi / 3.0
_MIRROR_HAT = np.array([np.cos(_MIRROR_ANGLE), np.sin(_MIRROR_ANGLE)])
_NORMAL_HAT = np.array([
    np.cos(_MIRROR_ANGLE - 0.5 * np.pi),
    np.sin(_MIRROR_ANGLE - 0.5 * np.pi),
])

# ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l. Matches the developed junction.
DEFAULT_JUNCTION = np.array([
    0.00000000e+00, 1.63798590e+00, 6.11348571e-01, 1.47929004e-01,
    1.94666468e+00, 5.90633914e-07, 1.00000011e+00, 4.99999973e+00,
])


def symmetric_gate_offsets(offsets):
    """Mirror a transport waveform about mid-path.

    Interior gates come in pairs. The gate after the midpoint is the negative
    of the gate before it, and the center gate does not move. The underlying
    ``(a, b)`` profile is already symmetric, so the ion then leaves the second
    half at the same speeds it used to enter the first.
    """
    values = np.asarray(offsets, dtype=float).reshape(-1).copy()
    half = values.size // 2
    for index in range(half):
        paired = 0.5 * (values[index] - values[-1 - index])
        values[index] = paired
        values[-1 - index] = -paired
    if values.size % 2 == 1:
        values[half] = 0.0
    return values


def bend_centerline(bend_x, handle, end_handle, cross_s, n=4000):
    """Quadratic from the arm tangent to the mirror tangent, and its normal.

    The control of the quadratic is the intersection of those two tangents, so
    the curve leaves the arm along +x and arrives along the +120 degree
    normal. ``cross_s`` is the only radius in this curve.
    """
    start = np.array([float(bend_x) + float(handle), 0.0])
    mirror_end = float(cross_s) * _MIRROR_HAT
    arrive = mirror_end - float(end_handle) * _NORMAL_HAT
    if abs(_NORMAL_HAT[1]) <= 1e-12:
        raise ValueError("mirror normal has no vertical component")
    length = arrive[1] / _NORMAL_HAT[1]
    control = arrive - length * _NORMAL_HAT
    t = np.linspace(0.0, 1.0, int(n))
    one_minus = 1.0 - t
    points = (
        (one_minus ** 2)[:, None] * start
        + (2.0 * one_minus * t)[:, None] * control
        + (t ** 2)[:, None] * arrive
    )
    derivative = (
        (2.0 * one_minus)[:, None] * (control - start)
        + (2.0 * t)[:, None] * (arrive - control)
    )
    tangent = derivative / np.maximum(np.linalg.norm(derivative, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    if float(np.dot(normal[len(normal) // 2], _NORMAL_HAT)) < 0.0:
        normal = -normal
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    return points, normal, arc


def _station_arc():
    """Arc-length fractions of the seven developed bend controls."""
    points, _normal, arc = bend_centerline(-1.0, 0.12, 0.048, _CROSS_S)
    total = float(arc[-1])
    fractions = []
    for station in _DEVELOPED_STATIONS:
        index = int(np.argmin(np.linalg.norm(points - station, axis=1)))
        fractions.append(float(arc[index]) / total)
    return np.asarray(fractions, dtype=float)


_STATION_ARC = _station_arc()


def interior_xy_from_s(cross_s, bend_x=-1.0, handle=0.12, end_handle=0.048):
    """Bend ``(x, y)`` fixed by the radius ``cross_s``.

    ``cross_s`` is the distance from the origin along the +120 degree line.
    It sets the tangent-matched curve, and the seven controls are the same
    arc stations used for the developed bend. Those coordinates are not free.
    ``cross_s = 0.1`` returns the developed controls.
    """
    points, _normal, arc = bend_centerline(bend_x, handle, end_handle, cross_s)
    distance = _STATION_ARC * float(arc[-1])
    return np.column_stack([
        np.interp(distance, arc, points[:, axis]) for axis in range(2)
    ])


def interior_xyz_from_sz(
    cross_s, height, bend_x=-1.0, handle=0.12, end_handle=0.048,
):
    """Bend control ``(x, y, z)``. ``(x, y)`` comes only from ``cross_s``."""
    height = np.asarray(height, dtype=float).reshape(-1)
    if height.size != N_INTERIOR:
        raise ValueError("bend heights must have length {}".format(N_INTERIOR))
    return np.column_stack([
        interior_xy_from_s(cross_s, bend_x, handle, end_handle),
        height,
    ])


def _feature_table():
    rows = [
        ("ai", -0.25, 0.35),
        ("da", 0.90, 2.30),
        ("bi_x", 0.25, 1.10),
        ("bi_y", 0.00, 0.45),
        ("bo_x", 1.30, 2.60),
        ("bo_y", 0.00, 0.35),
        ("ci_l", 0.60, 1.80),
        ("co_l", 3.50, 6.50),
        ("cross_s", 0.0, _CROSS_HI),
    ]
    for index in range(N_INTERIOR):
        rows.append(("c{}_z".format(index), _Z_LO, _Z_HI))
    rows.extend([
        ("curve_a", 1.20, 5.50),
        ("curve_b", 0.60, 2.80),
    ])
    return tuple(rows)


# name, lower, upper. Nominal values live on TransportDesign.
_BOUNDS = _feature_table()


@dataclass
class TransportDesign:
    """One Y junction and one round-trip trajectory through it.

    ``cross_s`` is the bend radius. It fixes the control ``(x, y)``. Those
    coordinates are not separate search parameters. ``interior_z`` is the
    height of each control. ``arm_xyz`` is fixed.
    The start and end stay at 50 um.
    """

    junction: np.ndarray
    interior_z: np.ndarray
    curves: tuple
    arm_xyz: np.ndarray = field(default_factory=lambda: np.array(_ARM, dtype=float))
    cross_s: float = _CROSS_S
    x_start: float = -5.0
    bend_x: float = -1.0
    handle: float = 0.12
    end_handle: float = 0.048
    arm_handle: float = 0.12
    n_time_gates: int = 8
    gate_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z_bend: float = ION_HEIGHT
    z_cross: float = ION_HEIGHT
    z_arm: float = field(init=False)
    interior_xyz: np.ndarray = field(init=False)

    def __post_init__(self):
        self.junction = np.asarray(self.junction, dtype=float).reshape(-1).copy()
        if self.junction.size != 8:
            raise ValueError("junction controls must be 8 values")
        self.interior_z = np.asarray(self.interior_z, dtype=float).reshape(-1).copy()
        if self.interior_z.size != N_INTERIOR:
            raise ValueError(
                "interior_z must have length {}".format(N_INTERIOR)
            )
        self.interior_xyz = interior_xyz_from_sz(
            self.cross_s, self.interior_z,
            self.bend_x, self.handle, self.end_handle,
        )
        self.arm_xyz = np.asarray(self.arm_xyz, dtype=float)
        if self.arm_xyz.shape != (N_ARM, 3):
            raise ValueError(
                "arm_xyz must have shape ({}, 3): x, y, and height "
                "of each arm control".format(N_ARM)
            )
        self.arm_xyz = self.arm_xyz.copy()
        if len(self.curves) != 2:
            raise ValueError("curves must be (a, b)")
        self.curves = (float(self.curves[0]), float(self.curves[1]))
        self.z_arm = float(ION_HEIGHT)
        self.z_bend = float(self.z_bend)
        self.z_cross = float(self.z_cross)
        self.arm_handle = float(self.arm_handle)
        if self.arm_handle <= 0.0:
            raise ValueError("arm_handle must be positive")
        self.cross_s = float(self.cross_s)
        self.x_start = float(self.x_start)
        self.bend_x = float(self.bend_x)
        self.handle = float(self.handle)
        self.end_handle = float(self.end_handle)
        if self.end_handle <= 0.0:
            raise ValueError("end_handle must be positive")
        if self.cross_s < 0.0:
            raise ValueError("cross_s must be non-negative")
        if not self.bend_x > self.x_start:
            raise ValueError("bend_x must be greater than x_start")
        heights = np.concatenate([
            self.arm_xyz[:, 2], self.interior_z, [self.z_bend, self.z_cross],
        ])
        if min(self.z_arm, float(heights.min())) <= 0.05:
            raise ValueError("ion height must stay above the chip")
        n_interior = int(self.n_time_gates) - 1
        if n_interior < 1:
            raise ValueError("n_time_gates must be at least 2")
        offsets = np.asarray(self.gate_offsets, dtype=float).reshape(-1)
        if offsets.size == 0:
            offsets = np.zeros(n_interior)
        if offsets.size != n_interior:
            raise ValueError(
                "gate_offsets must have length {}, one per interior time gate".format(n_interior)
            )
        self.gate_offsets = symmetric_gate_offsets(offsets)

    def __setattr__(self, name, value):
        if name == "gate_offsets":
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size:
                value = symmetric_gate_offsets(value)
        if name == "cross_s" and not isinstance(value, property):
            value = float(np.asarray(value, dtype=float).reshape(-1)[0])
        if name == "interior_z":
            value = np.asarray(value, dtype=float).reshape(-1).copy()
        object.__setattr__(self, name, value)
        if name in ("cross_s", "interior_z", "bend_x", "handle", "end_handle"):
            if hasattr(self, "cross_s") and hasattr(self, "interior_z"):
                try:
                    object.__setattr__(
                        self,
                        "interior_xyz",
                        interior_xyz_from_sz(
                            self.cross_s, self.interior_z,
                            getattr(self, "bend_x", -1.0),
                            getattr(self, "handle", 0.12),
                            getattr(self, "end_handle", 0.048),
                        ),
                    )
                except Exception:
                    pass


def within_transport_caps(design):
    """Bend radius ``cross_s`` and control heights stay within 15 um of the RF center."""
    if design.cross_s * L_SCALE > CROSS_CAP_M + 1e-12:
        return False
    heights = np.concatenate([
        design.arm_xyz[:, 2],
        design.interior_z,
        [design.z_bend, design.z_cross],
    ])
    deviation_m = np.abs(heights - ION_HEIGHT) * L_SCALE
    return bool(np.all(deviation_m <= HEIGHT_CAP_M + 1e-12))


class DesignSpace:
    """Bounds and packing for one :class:`TransportDesign`."""

    def __init__(self):
        self.names = tuple(row[0] for row in _BOUNDS)
        self.lower = np.array([row[1] for row in _BOUNDS], dtype=float)
        self.upper = np.array([row[2] for row in _BOUNDS], dtype=float)
        self.x_start = -5.0
        self.bend_x = -1.0
        self.handle = 0.12
        self.end_handle = 0.048
        self.arm_handle = 0.12

    @property
    def dim(self):
        return len(self.names)

    def nominal_design(self):
        """Developed junction. Radius ``cross_s`` is 0.1; height stays at 50 um."""
        return TransportDesign(
            junction=DEFAULT_JUNCTION.copy(),
            interior_z=np.full(N_INTERIOR, ION_HEIGHT),
            curves=(3.0, 1.5),
            arm_xyz=np.array(_ARM, dtype=float),
            cross_s=_CROSS_S,
            x_start=self.x_start,
            bend_x=self.bend_x,
            handle=self.handle,
            end_handle=self.end_handle,
            arm_handle=self.arm_handle,
            z_bend=ION_HEIGHT,
            z_cross=ION_HEIGHT,
        )

    def nominal(self):
        return self.pack(self.nominal_design())

    def pack(self, design):
        values = np.concatenate([
            np.asarray(design.junction, dtype=float).reshape(8),
            [float(design.cross_s)],
            np.asarray(design.interior_z, dtype=float).reshape(N_INTERIOR),
            [design.curves[0], design.curves[1]],
        ])
        if values.size != self.dim:
            raise ValueError("packed design has the wrong length")
        return values

    def unpack(self, vector):
        x = np.asarray(vector, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(
                "design vector has length {}, expected {}".format(x.size, self.dim)
            )
        z0 = 9
        return TransportDesign(
            junction=x[:8],
            interior_z=x[z0:z0 + N_INTERIOR],
            curves=(x[-2], x[-1]),
            arm_xyz=np.array(_ARM, dtype=float),
            cross_s=x[8],
            x_start=self.x_start,
            bend_x=self.bend_x,
            handle=self.handle,
            end_handle=self.end_handle,
            arm_handle=self.arm_handle,
            z_bend=ION_HEIGHT,
            z_cross=ION_HEIGHT,
        )

    def clip(self, vector):
        return np.clip(np.asarray(vector, dtype=float), self.lower, self.upper)

    def to_unit(self, vector):
        x = np.asarray(vector, dtype=float)
        span = self.upper - self.lower
        return (x - self.lower) / span

    def from_unit(self, unit):
        u = np.asarray(unit, dtype=float)
        return self.lower + u * (self.upper - self.lower)

    def sample(self, rng, n, fraction=1.0, cap_fraction=None):
        """Uniform samples in a box shrunk toward the nominal design.

        ``fraction`` 1 uses the full bounds. Smaller values stay closer to
        the developed junction and are more likely to confine. ``cap_fraction``
        overrides that shrink for the bend radius and the control heights,
        whose full bounds are the 15 um RF-center caps.
        """
        lo, hi = self._bounds(fraction, cap_fraction)
        unit = rng.random((int(n), self.dim))
        return lo + unit * (hi - lo)

    def sample_latin(self, rng, n, fraction=1.0, cap_fraction=None):
        """Space-filling samples. Each coordinate is split into ``n`` equal bins.

        One design falls in each bin, so the radius, the heights, the
        junction, and ``(a, b)`` cover their ranges evenly instead of clumping.
        """
        from scipy.stats.qmc import LatinHypercube

        lo, hi = self._bounds(fraction, cap_fraction)
        unit = LatinHypercube(d=self.dim, seed=rng).random(int(n))
        return lo + unit * (hi - lo)

    def _bounds(self, fraction, cap_fraction):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")
        if cap_fraction is None:
            cap_fraction = fraction
        if not 0.0 <= cap_fraction <= 1.0:
            raise ValueError("cap_fraction must be between 0 and 1")
        center = self.nominal()
        lo = center + fraction * (self.lower - center)
        hi = center + fraction * (self.upper - center)
        for index, name in enumerate(self.names):
            if name == "cross_s" or name.endswith("_z"):
                lo[index] = center[index] + cap_fraction * (self.lower[index] - center[index])
                hi[index] = center[index] + cap_fraction * (self.upper[index] - center[index])
        return lo, hi


class BoxGaussian:
    """Diagonal Gaussian on the normalized design box, clipped to the bounds."""

    def __init__(self, unit_mean, unit_std, space):
        self.unit_mean = np.asarray(unit_mean, dtype=float).reshape(-1).copy()
        self.unit_std = np.asarray(unit_std, dtype=float).reshape(-1).copy()
        self.space = space
        if self.unit_mean.size != space.dim or self.unit_std.size != space.dim:
            raise ValueError("Gaussian dimension does not match the design space")

    @classmethod
    def around(cls, space, mean_x=None, std=0.15):
        if mean_x is None:
            mean_x = space.nominal()
        unit = np.clip(space.to_unit(mean_x), 0.0, 1.0)
        return cls(unit, np.full(space.dim, float(std)), space)

    def sample(self, rng, n):
        unit = rng.normal(self.unit_mean, self.unit_std, size=(int(n), self.space.dim))
        unit = np.clip(unit, 0.0, 1.0)
        return self.space.from_unit(unit)

    def update(self, elite_x, smooth=0.7, std_floor=0.05, std_cap=0.45):
        """Move toward the elite cloud. ``smooth`` 1 replaces the old Gaussian."""
        if len(elite_x) == 0:
            return
        unit = np.clip(self.space.to_unit(np.asarray(elite_x, dtype=float)), 0.0, 1.0)
        mu = unit.mean(axis=0)
        std = unit.std(axis=0)
        std = np.maximum(std, std_floor)
        self.unit_mean = smooth * mu + (1.0 - smooth) * self.unit_mean
        mixed = smooth * std + (1.0 - smooth) * self.unit_std
        self.unit_std = np.clip(mixed, std_floor, std_cap)
