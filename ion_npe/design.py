"""Junction and trajectory design vector.

The search coordinates are the eight symmetric Y-junction controls, the arm
and bend controls ``(x, y, z)``, the height where the bend starts, the height
at the crossing, the ``(a, b)`` time-profile parameters, and the crossing
distance. The ion starts and ends at 50 um. Between those endpoints, including
at the crossing, the path may leave 50 um. Controls are spaced more tightly
near the junction.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import CROSS_CAP_M, HEIGHT_CAP_M, ION_HEIGHT, L_SCALE

# (nominal x, x lower, x upper, nominal y, y lower, y upper).
# Arm controls sit on the straight section, after the fixed 50 um start and
# before the bend. Their x intervals are disjoint. y may leave the axis.
_ARM = (
    (-4.50, -4.75, -4.25, 0.0, -0.06, 0.06),
    (-3.55, -3.90, -3.20, 0.0, -0.08, 0.08),
    (-2.60, -2.95, -2.25, 0.0, -0.10, 0.10),
    (-1.70, -2.00, -1.40, 0.0, -0.12, 0.12),
)
N_ARM = len(_ARM)

# Bend controls. The x intervals are disjoint and shorten toward the junction.
# The last control sits just before the junction handle, so the polygon keeps
# closing up through the two squares at the crossing.
_INTERIOR = (
    (-0.7263, -0.780, -0.680, 0.0009, 0.00, 0.06),
    (-0.5925, -0.650, -0.550, 0.0034, 0.00, 0.09),
    (-0.4762, -0.530, -0.440, 0.0076, 0.00, 0.12),
    (-0.3751, -0.420, -0.340, 0.0133, 0.00, 0.16),
    (-0.2874, -0.320, -0.260, 0.0207, 0.00, 0.20),
    (-0.2114, -0.240, -0.190, 0.0302, 0.00, 0.26),
    (-0.1459, -0.175, -0.130, 0.0429, 0.00, 0.32),
)
N_INTERIOR = len(_INTERIOR)
_Z_SPAN = HEIGHT_CAP_M / L_SCALE
_Z_LO = ION_HEIGHT - _Z_SPAN
_Z_HI = ION_HEIGHT + _Z_SPAN
_CROSS_HI = CROSS_CAP_M / L_SCALE

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
    ]
    for index, (_x, x_lo, x_hi, _y, y_lo, y_hi) in enumerate(_ARM):
        rows.append(("a{}_x".format(index), x_lo, x_hi))
        rows.append(("a{}_y".format(index), y_lo, y_hi))
        rows.append(("a{}_z".format(index), _Z_LO, _Z_HI))
    for index, (_x, x_lo, x_hi, _y, y_lo, y_hi) in enumerate(_INTERIOR):
        rows.append(("c{}_x".format(index), x_lo, x_hi))
        rows.append(("c{}_y".format(index), y_lo, y_hi))
        rows.append(("c{}_z".format(index), _Z_LO, _Z_HI))
    rows.extend([
        ("bend_z", _Z_LO, _Z_HI),
        ("cross_z", _Z_LO, _Z_HI),
        ("curve_a", 1.20, 5.50),
        ("curve_b", 0.60, 2.80),
        ("cross_s", 0.06, _CROSS_HI),
    ])
    return tuple(rows)


# name, lower, upper. Nominal values live on TransportDesign.
_BOUNDS = _feature_table()


@dataclass
class TransportDesign:
    """One Y junction and one round-trip trajectory through it.

    ``arm_xyz`` and ``interior_xyz`` have shape ``(N, 3)``. The third column
    is the ion height at that control, in trap heights. ``z_arm`` is the fixed
    height at the start and at the end. ``z_bend`` is the height where the
    arm meets the bend. ``z_cross`` is the height at the crossing.
    """

    junction: np.ndarray
    arm_xyz: np.ndarray
    interior_xyz: np.ndarray
    curves: tuple
    cross_s: float
    x_start: float = -5.0
    bend_x: float = -1.0
    handle: float = 0.12
    end_handle: float = 0.048
    arm_handle: float = 0.12
    n_time_gates: int = 6
    gate_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z_bend: float = ION_HEIGHT
    z_cross: float = ION_HEIGHT
    z_arm: float = field(init=False)

    def __post_init__(self):
        self.junction = np.asarray(self.junction, dtype=float).reshape(-1).copy()
        if self.junction.size != 8:
            raise ValueError("junction controls must be 8 values")
        self.arm_xyz = np.asarray(self.arm_xyz, dtype=float)
        if self.arm_xyz.shape != (N_ARM, 3):
            raise ValueError(
                "arm_xyz must have shape ({}, 3): x, y, and height "
                "of each arm control".format(N_ARM)
            )
        self.arm_xyz = self.arm_xyz.copy()
        self.interior_xyz = np.asarray(self.interior_xyz, dtype=float)
        if self.interior_xyz.shape != (N_INTERIOR, 3):
            raise ValueError(
                "interior_xyz must have shape ({}, 3): x, y, and height "
                "of each bend control".format(N_INTERIOR)
            )
        self.interior_xyz = self.interior_xyz.copy()
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
            self.arm_xyz[:, 2], self.interior_xyz[:, 2], [self.z_bend, self.z_cross],
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
        object.__setattr__(self, name, value)


def within_transport_caps(design):
    """Crossing distance and every free height stay within 10 um of the standard."""
    if design.cross_s * L_SCALE > CROSS_CAP_M + 1e-12:
        return False
    heights = np.concatenate([
        design.arm_xyz[:, 2],
        design.interior_xyz[:, 2],
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
        """Developed junction. The path starts at 50 um and may leave it later."""
        arm = np.array([[row[0], row[3], ION_HEIGHT] for row in _ARM])
        interior = np.array([[row[0], row[3], ION_HEIGHT] for row in _INTERIOR])
        return TransportDesign(
            junction=DEFAULT_JUNCTION.copy(),
            arm_xyz=arm,
            interior_xyz=interior,
            curves=(3.0, 1.5),
            cross_s=0.1,
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
            np.asarray(design.arm_xyz, dtype=float).reshape(-1),
            np.asarray(design.interior_xyz, dtype=float).reshape(-1),
            [
                design.z_bend, design.z_cross,
                design.curves[0], design.curves[1], design.cross_s,
            ],
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
        n_arm = N_ARM * 3
        n_xyz = N_INTERIOR * 3
        cursor = 8 + n_arm
        return TransportDesign(
            junction=x[:8],
            arm_xyz=x[8:cursor].reshape(N_ARM, 3),
            interior_xyz=x[cursor:cursor + n_xyz].reshape(N_INTERIOR, 3),
            z_bend=x[cursor + n_xyz],
            z_cross=x[cursor + n_xyz + 1],
            curves=(x[cursor + n_xyz + 2], x[cursor + n_xyz + 3]),
            cross_s=x[cursor + n_xyz + 4],
            x_start=self.x_start,
            bend_x=self.bend_x,
            handle=self.handle,
            end_handle=self.end_handle,
            arm_handle=self.arm_handle,
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
        overrides that shrink for the bend heights and the crossing distance,
        whose full bounds are the 10 um caps.
        """
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
        unit = rng.random((int(n), self.dim))
        return lo + unit * (hi - lo)

    def sample_latin(self, rng, n, fraction=1.0, cap_fraction=None):
        """Space-filling samples. Each coordinate is split into ``n`` equal bins.

        One design falls in each bin, so the height, crossing, junction, and
        bend controls cover their ranges evenly instead of clumping.
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
