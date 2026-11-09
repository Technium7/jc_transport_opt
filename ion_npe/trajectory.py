"""Three-dimensional transport path and the time law that moves along it.

The ion starts at ``(x_start, 0, z_arm)`` and ends at the reflection of that
point, at the same height. Both endpoints are 50 um. The arm stays on the
developed path. The bend radius ``cross_s`` fixes the bend ``(x, y)``. Height
is the free bend control. ``z`` is taken from the design. It is not solved
from a height model.
"""

import numpy as np
from scipy.interpolate import BSpline, CubicSpline

from .constants import HEIGHT_CAP_M, ION_HEIGHT, L_SCALE
from .design import TransportDesign


def generate_knot_vector(num_ctrlpts, degree=3):
    """Open uniform knot vector, matching the developed spline code."""
    n = num_ctrlpts - 1
    knots = [0.0] * (degree + 1)
    middle = [i / (n - degree + 1) for i in range(1, n - degree + 1)]
    knots.extend(middle)
    knots.extend([1.0] * (degree + 1))
    return np.asarray(knots, dtype=float)

MIRROR_ANGLE = 2.0 * np.pi / 3.0
MIRROR_HAT = np.array([np.cos(MIRROR_ANGLE), np.sin(MIRROR_ANGLE)])
NORMAL_HAT = np.array([
    np.cos(MIRROR_ANGLE - 0.5 * np.pi),
    np.sin(MIRROR_ANGLE - 0.5 * np.pi),
])


def reflect_xy(points):
    """Reflect xy points across the line through the origin at +120 degrees."""
    points = np.asarray(points, dtype=float)
    proj = np.sum(points * MIRROR_HAT, axis=-1, keepdims=True)
    return 2.0 * proj * MIRROR_HAT - points


def end_point(x_start):
    """Image of the inbound start. The outbound leg finishes here."""
    return reflect_xy(np.array([x_start, 0.0]))


def _rho(x, a, b):
    zeta = np.exp(a * b * (2.0 * x - 1.0))
    w = (zeta - 1j * np.exp(-a)) / (zeta - 1j * np.exp(a))
    return np.log(np.abs(w))


def profile_func(x, a, b):
    """Smooth step from 0 at x = 0 to 1 at x = 1."""
    return (_rho(x, a, b) - _rho(0.0, a, b)) / (_rho(1.0, a, b) - _rho(0.0, a, b))


def _spline_basis(tau):
    return np.array([1.0, tau, tau * tau, tau ** 3], dtype=float)


def _spline_dbasis(tau):
    return np.array([0.0, 1.0, 2.0 * tau, 3.0 * tau * tau], dtype=float)


def _block_diag(blocks):
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size))
    start = 0
    for block in blocks:
        k = block.shape[0]
        out[start:start + k, start:start + k] = block
        start += k
    return out


def _solve_spline_1d(x0, gates, dt):
    """Cubic coefficients. Position matches the gates; velocity is zero at the ends."""
    n_segments = len(dt)
    n_coeffs = 4 * n_segments

    def segment_row(index, basis):
        row = np.zeros(n_coeffs)
        row[4 * index:4 * index + 4] = basis
        return row

    rows = [segment_row(0, _spline_basis(0.0))]
    rhs = [float(x0)]
    rows.append(segment_row(0, _spline_dbasis(0.0)))
    rhs.append(0.0)
    rows.append(segment_row(n_segments - 1, _spline_dbasis(dt[-1])))
    rhs.append(0.0)
    for i in range(n_segments):
        rows.append(segment_row(i, _spline_basis(dt[i])))
        rhs.append(float(gates[i]))
    for i in range(n_segments - 1):
        rows.append(
            segment_row(i, _spline_basis(dt[i])) - segment_row(i + 1, _spline_basis(0.0))
        )
        rhs.append(0.0)
        rows.append(
            segment_row(i, _spline_dbasis(dt[i]))
            - segment_row(i + 1, _spline_dbasis(0.0))
        )
        rhs.append(0.0)
    constraint = np.stack(rows)
    target = np.asarray(rhs, dtype=float)
    blocks = []
    for duration in dt:
        blocks.append(np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0 * duration, 6.0 * duration ** 2],
            [0.0, 0.0, 6.0 * duration ** 2, 12.0 * duration ** 3],
        ]))
    cost = _block_diag(blocks)
    n_constraints = constraint.shape[0]
    kkt = np.block([
        [2.0 * cost, constraint.T],
        [constraint, np.zeros((n_constraints, n_constraints))],
    ])
    kkt_rhs = np.concatenate([np.zeros(n_coeffs), target])
    return np.linalg.solve(kkt, kkt_rhs)[:n_coeffs]


def _eval_spline_1d(coeffs, times, dt):
    times = np.asarray(times, dtype=float)
    cumulative = np.cumsum(dt)
    segment = np.clip(np.searchsorted(cumulative, times, side="right"), 0, len(dt) - 1)
    starts = np.concatenate([[0.0], cumulative[:-1]])
    local = times - starts[segment]
    values = np.empty(times.shape)
    for i in range(len(dt)):
        chosen = segment == i
        if not np.any(chosen):
            continue
        c0, c1, c2, c3 = coeffs[4 * i:4 * i + 4]
        t = local[chosen]
        values[chosen] = c0 + c1 * t + c2 * t * t + c3 * t ** 3
    return values


def time_gate_samples(curves, n_gates=8, gate_offsets=None):
    """Arc-fraction samples of the transport waveform, including the fixed ends.

    ``gate_offsets`` moves the interior samples, the same freedom as the
    intermediate gates in the straight-line transport search. The ends stay
    at 0 and 1, and the spline still leaves and arrives at rest. The interior
    gates are equally spaced and one sits at mid-path. A symmetric waveform
    uses the four gates from the start through that center. The center offset
    stays at zero, and the gates after it mirror the first three.
    """
    a, b = curves
    knots = np.linspace(0.0, 1.0, int(n_gates) + 1)
    samples = np.real(profile_func(knots, a, b)).astype(float).copy()
    if gate_offsets is not None:
        offsets = np.asarray(gate_offsets, dtype=float).reshape(-1)
        if offsets.size != int(n_gates) - 1:
            raise ValueError(
                "gate offsets must have length {}, one per interior time gate".format(
                    int(n_gates) - 1
                )
            )
        samples[1:-1] = samples[1:-1] + offsets
    return knots, samples


def one_sided_time_gates(curves, n_gates=8, gate_offsets=None):
    """Interior gates on the first half, including the center gate.

    Returns ``(tau, arc fraction)``. The last gate is at mid-path. Gates on
    the return half are the mirrors of the first three and are not included.
    """
    knots, samples = time_gate_samples(curves, n_gates, gate_offsets)
    center = 1 + (samples.size - 2) // 2
    return knots[1:center + 1], samples[1:center + 1]


def s_of_tau(tau, curves, n_gates=8, gate_offsets=None):
    """Arc-length fraction versus tau. Velocity is zero at tau = 0 and tau = 1."""
    knots, samples = time_gate_samples(curves, n_gates, gate_offsets)
    dt = np.diff(knots)
    coeffs = _solve_spline_1d(samples[0], samples[1:], dt)
    return _eval_spline_1d(coeffs, tau, dt)


def _require_design(design):
    if not isinstance(design, TransportDesign):
        raise TypeError("expected a TransportDesign")


def arm_polygon(design):
    """Clamped controls of the inbound arm.

    The path leaves the start at 50 um, tangent to the arm. It arrives at the
    bend entrance at ``z_bend``, with a flat vertical tangent. The arm
    controls stay on the developed RF-null path.
    """
    _require_design(design)
    start = np.array([design.x_start, 0.0, design.z_arm])
    depart = np.array([design.x_start + design.arm_handle, 0.0, design.z_arm])
    arrive = np.array([design.bend_x - design.arm_handle, 0.0, design.z_bend])
    join = np.array([design.bend_x, 0.0, design.z_bend])
    return np.vstack([start, depart, design.arm_xyz, arrive, join])


def control_polygon(design):
    """Clamped ``(x, y, z)`` controls of the inbound bend.

    The curve leaves the arm at ``z_bend`` and arrives at ``z_cross``.
    ``handle`` sets the tangent on the arm. ``end_handle`` is shorter, so the
    two squares at the crossing continue the tightening of the free controls.
    Interior ``(x, y)`` is the tangent-matched bend at radius ``cross_s``.
    Those coordinates stay on that curve. The crossing stays on the +120
    degree line so the return path is its reflection.
    """
    _require_design(design)
    p_cross_xy = design.cross_s * MIRROR_HAT
    p0 = np.array([design.bend_x, 0.0, design.z_bend])
    p1 = np.array([design.bend_x + design.handle, 0.0, design.z_bend])
    p_cross = np.array([p_cross_xy[0], p_cross_xy[1], design.z_cross])
    p_prev = np.array([
        p_cross_xy[0] - design.end_handle * NORMAL_HAT[0],
        p_cross_xy[1] - design.end_handle * NORMAL_HAT[1],
        design.z_cross,
    ])
    return np.vstack([p0, p1, design.interior_xyz, p_prev, p_cross])


def clamped_spline(ctrl, n_sample=500):
    """Cubic B-spline of ``ctrl`` as a function of arc length, and its length."""
    ctrl = np.asarray(ctrl, dtype=float)
    if float(np.max(np.ptp(ctrl[:, :2], axis=0))) <= 1e-12:
        point = np.array([ctrl[0, 0], ctrl[0, 1], float(np.mean(ctrl[:, 2]))])

        def at_point(distance):
            distance = np.asarray(distance, dtype=float)
            return np.broadcast_to(point, np.shape(distance) + (3,)).copy()

        return at_point, 0.0
    degree = 3
    knots = generate_knot_vector(len(ctrl), degree)
    splines = [BSpline(knots, ctrl[:, k], degree) for k in range(ctrl.shape[1])]
    u = np.linspace(0.0, 1.0, n_sample)
    pts = np.column_stack([spline(u) for spline in splines])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return CubicSpline(arc, pts), float(arc[-1])


class InboundPath:
    """Straight arm from the start, then the junction bend."""

    def __init__(self, arm_of_s, arm_length, bend_of_s, bend_length):
        self.arm_of_s = arm_of_s
        self.arm_length = float(arm_length)
        self.bend_of_s = bend_of_s
        self.bend_length = float(bend_length)

    @property
    def length(self):
        return self.arm_length + self.bend_length

    def points(self, distance):
        distance = np.asarray(distance, dtype=float)
        out = np.empty(distance.shape + (3,))
        on_arm = distance <= self.arm_length
        out[on_arm] = self.arm_of_s(np.clip(distance[on_arm], 0.0, self.arm_length))
        if self.bend_length > 0.0:
            out[~on_arm] = self.bend_of_s(
                np.clip(distance[~on_arm] - self.arm_length, 0.0, self.bend_length)
            )
        elif np.any(~on_arm):
            out[~on_arm] = self.arm_of_s(self.arm_length)
        return out


def make_inbound(design):
    """Inbound geometry. The arm and the bend are separate clamped splines."""
    arm = arm_polygon(design)
    bend = control_polygon(design)
    arm_of_s, arm_length = clamped_spline(arm)
    bend_of_s, bend_length = clamped_spline(bend)
    return InboundPath(arm_of_s, arm_length, bend_of_s, bend_length), bend


def _round_trip_at(inbound, design, distance_dimless):
    """Points at dimensionless distances along the round trip."""
    total = 2.0 * inbound.length
    distance = np.asarray(distance_dimless, dtype=float)
    on_outbound = distance > inbound.length
    xyz = inbound.points(np.where(on_outbound, total - distance, distance))
    xyz = np.array(xyz, copy=True)
    xyz[on_outbound, :2] = reflect_xy(xyz[on_outbound, :2])
    return xyz


def controls_stay_ordered(design):
    """Arm and bend control x both increase, so neither polygon folds."""
    arm_ordered = np.all(np.diff(arm_polygon(design)[:, 0]) > 1e-4)
    bend_ordered = np.all(np.diff(control_polygon(design)[:, 0]) > 1e-4)
    return bool(arm_ordered and bend_ordered)


def height_within_cap(design, n=241):
    """The ion stays within 15 um of the 50 um height along the whole path."""
    path = path_geometry(design, n=int(n))
    deviation_m = float(np.max(np.abs(path[:, 2] - ION_HEIGHT))) * L_SCALE
    return deviation_m <= HEIGHT_CAP_M + 1e-9


def path_geometry(design, n=601):
    """Round-trip samples spaced uniformly in arc length. Shape ``(n, 3)``."""
    inbound, _ctrl = make_inbound(design)
    if inbound.length <= 0.0:
        raise ValueError("transport path has zero length")
    distance = np.linspace(0.0, 2.0 * inbound.length, int(n))
    return _round_trip_at(inbound, design, distance)


def dynamics_grid(design, speed_m_s, l_scale, sim_dt):
    """Round trip on the integration grid.

    Duration is path length / ``speed_m_s``. The outbound half is the inbound
    path reflected across +120 degrees. Height is not reflected.
    """
    if speed_m_s <= 0.0:
        raise ValueError("transport speed must be positive")
    inbound, ctrl = make_inbound(design)
    length_dimless = 2.0 * inbound.length
    if length_dimless <= 0.0:
        raise ValueError("transport path has zero length")
    length_m = length_dimless * l_scale
    n_steps = max(int(round(length_m / speed_m_s / sim_dt)), 1)
    tau = np.arange(n_steps + 1) / float(n_steps)
    s_prof = s_of_tau(tau, design.curves, design.n_time_gates, design.gate_offsets)
    xyz = _round_trip_at(inbound, design, s_prof * length_dimless)
    return {
        "xyz": xyz,
        "tau": tau,
        "n_steps": n_steps,
        "length_m": length_m,
        "duration_s": n_steps * sim_dt,
        "controls": ctrl,
    }
