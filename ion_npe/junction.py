"""Y-junction electrode geometry and the RF pseudopotential on it.

The eight controls are the symmetric spline used by the developed junction
optimizer: ``ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l``. The RF rails are
that spline. The DC electrodes follow the segmented Y junction: a center
triangle, inner electrodes along each arm, and a tilt electrode plus an
outer electrode on each side of each RF rail.
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
    ION_HEIGHT,
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

# Segmented DC layout, in trap heights. This is the Y junction with a 50 um
# ion height: 2.5 um RF gap, 2 um DC gap, 300 um arms, five inner electrodes
# on each arm, then 40 um tilt electrodes and 200 um outer electrodes.
N_INNER = 5
RF_GAP = 2.5 / 50.0
DC_GAP = 2.0 / 50.0
CUTOFF_LENGTH = 300.0 / 50.0
OUTER_TILT = 40.0 / 50.0
OUTER_LENGTH = 200.0 / 50.0
_ARM_ANGLE = np.pi / 3.0


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


def dc_electrode_names(n_inner=N_INNER):
    """Names of the segmented DC electrodes, in the order ``build_junction`` uses.

    ``dc_inner_{arm}_{index}`` lies on arm ``arm``. Index 0 is the electrode
    nearest the center. Arm 0 is the negative-x arm, and arms 1 and 2 are
    that arm rotated by 120 and 240 degrees. ``dc_tilt`` and ``dc_outer``
    are split along the RF rail: side 0 is the upper side of the rail before
    that rotation, side 1 the lower side. Rail 0 bulges toward +x.
    """
    n_inner = int(n_inner)
    names = ["dc_center"]
    for arm in range(3):
        for index in range(n_inner):
            names.append("dc_inner_{}_{}".format(arm, index))
    for kind in ("dc_tilt", "dc_outer"):
        for rail in range(3):
            for side in range(2):
                names.append("{}_{}_{}".format(kind, rail, side))
    return tuple(names)


def _arm_offset_point(distance, lateral):
    """Point on the +60 degree side of the +x rail.

    ``distance`` runs along the arm. ``lateral`` is the offset toward the
    rail, positive on the +x side of the arm.
    """
    return np.array([
        np.cos(_ARM_ANGLE) * distance + np.sin(_ARM_ANGLE) * lateral,
        np.sin(_ARM_ANGLE) * distance - np.cos(_ARM_ANGLE) * lateral,
    ])


def _signed_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_polygon(poly, tol=1e-10):
    """Drop a repeated closing vertex and consecutive duplicates."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) >= 2 and np.linalg.norm(poly[0] - poly[-1]) <= tol:
        poly = poly[:-1]
    if len(poly) == 0:
        return poly.reshape(0, 2)
    keep = [0]
    for index in range(1, len(poly)):
        if np.linalg.norm(poly[index] - poly[keep[-1]]) > tol:
            keep.append(index)
    return poly[np.asarray(keep)]


def _ccw(poly):
    poly = _clean_polygon(poly)
    if len(poly) >= 3 and _signed_area(poly) < 0.0:
        poly = poly[::-1]
    return poly


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-15:
        return np.zeros(2)
    return np.asarray(vector, dtype=float) / norm


def _polyline_distance(points, curve):
    """Distance from each point to an open polyline."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    start = curve[:-1]
    stop = curve[1:]
    edge = stop - start
    length2 = np.sum(edge * edge, axis=1)
    relative = points[:, None, :] - start[None, :, :]
    fraction = np.sum(relative * edge[None, :, :], axis=2) / np.maximum(length2, 1e-30)
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[:, :, None] * edge[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def _rotate_vector(vector, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    ])


def _line_intersection(origin_a, direction_a, origin_b, direction_b):
    matrix = np.column_stack([direction_a, -direction_b])
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(determinant) < 1e-14:
        return None
    try:
        parameter, _other = np.linalg.solve(matrix, origin_b - origin_a)
    except np.linalg.LinAlgError:
        return None
    return origin_a + parameter * direction_a


def _offset_round(curve, distance, arc_step=np.pi / 30.0):
    """Boundary of a constant gap. Positive ``distance`` is to the left of travel.

    A parallel curve folds where the rail bends tighter than the gap, and the
    fold is the spur in the tilt-electrode outline. Convex corners are drawn
    as circular arcs of radius ``distance``. On the concave side the fold is
    cut off, and anything that still falls inside the gap is dropped.
    """
    if distance < 0.0:
        return _offset_round(curve[::-1], -distance, arc_step)[::-1]
    if distance == 0.0:
        return _clean_polygon(curve)
    curve = _clean_polygon(np.asarray(curve, dtype=float), tol=1e-9)
    if len(curve) < 2:
        raise ValueError("offset curve needs at least two points")
    tangents = np.diff(curve, axis=0)
    normals = np.asarray(
        [[-direction[1], direction[0]] for direction in (_unit(tangent) for tangent in tangents)],
        dtype=float,
    )
    raw = [curve[0] + distance * normals[0]]
    for index in range(len(normals) - 1):
        left = normals[index]
        right = normals[index + 1]
        turn = float(np.arctan2(
            left[0] * right[1] - left[1] * right[0],
            np.clip(np.dot(left, right), -1.0, 1.0),
        ))
        vertex = curve[index + 1]
        if turn < -np.radians(1.0):
            incoming = vertex + distance * left
            if np.linalg.norm(incoming - raw[-1]) > 1e-9:
                raw.append(incoming)
            steps = max(1, int(np.ceil(abs(turn) / arc_step)))
            for step in range(1, steps + 1):
                raw.append(vertex + distance * _rotate_vector(left, turn * step / steps))
        elif turn > np.radians(1.0):
            hit = _line_intersection(
                vertex + distance * left,
                tangents[index],
                vertex + distance * right,
                tangents[index + 1],
            )
            if hit is not None and np.linalg.norm(hit - vertex) <= 6.0 * distance:
                if np.linalg.norm(hit - raw[-1]) > 1e-9:
                    raw.append(hit)
            else:
                raw.append(vertex + distance * left)
                raw.append(vertex + distance * right)
        else:
            outgoing = vertex + distance * right
            if np.linalg.norm(outgoing - raw[-1]) > 1e-9:
                raw.append(outgoing)
    end = curve[-1] + distance * normals[-1]
    if np.linalg.norm(end - raw[-1]) > 1e-9:
        raw.append(end)
    raw = np.vstack(raw)
    # Points on the fold sit closer than the gap to some other part of the rail.
    outside = _polyline_distance(raw, curve) >= distance * (1.0 - 1e-3)
    outside[0] = True
    outside[-1] = True
    points = []
    for point, kept in zip(raw, outside):
        if kept and (not points or np.linalg.norm(point - points[-1]) > 1e-9):
            points.append(point)
    changed = True
    while changed:
        changed = False
        index = 1
        while index < len(points) - 1:
            incoming = points[index] - points[index - 1]
            outgoing = points[index + 1] - points[index]
            bridge = 0.5 * (points[index - 1] + points[index + 1])
            if np.dot(incoming, outgoing) < 0.0 and float(
                _polyline_distance(bridge, curve)[0]
            ) >= distance * (1.0 - 1e-3):
                del points[index]
                changed = True
                continue
            index += 1
    polygon = _clean_polygon(np.vstack(points), tol=1e-9)
    if len(polygon) < 2:
        raise ValueError("constant-gap offset left the curve")
    return polygon


def _offset_away(curve, distance, other):
    """Offset ``curve`` away from ``other`` by ``distance``.

    The sign is chosen along the curve, not only at its middle. The middle of
    a rail is the tip, where the spline tangent is not a reliable normal.
    """
    curve = np.asarray(curve, dtype=float)
    other = np.asarray(other, dtype=float)
    votes = []
    stations = (
        len(curve) // 6,
        len(curve) // 3,
        len(curve) // 2,
        2 * len(curve) // 3,
        5 * len(curve) // 6,
    )
    for index in stations:
        tangent = curve[min(index + 1, len(curve) - 1)] - curve[max(index - 1, 0)]
        if np.linalg.norm(tangent) < 1e-8:
            continue
        normal = _unit([-tangent[1], tangent[0]])
        origin = curve[index]
        gap_left = float(np.min(np.linalg.norm(other - (origin + distance * normal), axis=1)))
        gap_right = float(np.min(np.linalg.norm(other - (origin - distance * normal), axis=1)))
        votes.append(1.0 if gap_left >= gap_right else -1.0)
    if not votes:
        raise ValueError("gap offset has no usable tangent")
    sign = 1.0 if sum(votes) >= 0.0 else -1.0
    return _offset_round(curve, sign * distance)


def _ray_radius(curve, direction):
    """Closest r > 0 where the ray ``r * direction`` meets ``curve``."""
    best = np.inf
    direction = np.asarray(direction, dtype=float)
    for start, stop in zip(curve[:-1], curve[1:]):
        matrix = np.column_stack([stop - start, -direction])
        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        if abs(det) < 1e-14:
            continue
        try:
            fraction, radius = np.linalg.solve(matrix, -start)
        except np.linalg.LinAlgError:
            continue
        if -1e-8 <= fraction <= 1.0 + 1e-8 and radius > 1e-8:
            best = min(best, float(radius))
    return best


def _extend_curve(spline, arm_end, lateral):
    """Straight arm ends at ``arm_end`` on both sides of a top-to-bottom spline."""
    start = _arm_offset_point(arm_end, lateral)
    end = start * np.array([1.0, -1.0])
    parts = []
    if np.linalg.norm(start - spline[0]) > 1e-8:
        parts.append(start.reshape(1, 2))
    parts.append(spline)
    if np.linalg.norm(end - spline[-1]) > 1e-8:
        parts.append(end.reshape(1, 2))
    return np.vstack(parts)



def _point_clear(point, rails, rf_gap):
    """True when ``point`` is outside every rail by at least ``rf_gap``."""
    sample = np.asarray(point, dtype=float).reshape(1, 2)
    for rail in rails:
        distance, inside = _rail_distance(sample, rail)
        if bool(inside[0]) or float(distance[0]) < rf_gap:
            return False
    return True


def _channel_apertures(rails, center_len, rf_gap):
    """Arm direction and the lateral reach of the channel at ``center_len``.

    Each side is the largest offset that still leaves one RF gap, so the
    center electrode meets whatever rail this design produced.
    """
    apertures = []
    for turn in range(3):
        angle = np.pi + turn * 2.0 * np.pi / 3.0
        arm = np.array([np.cos(angle), np.sin(angle)])
        perp = np.array([-arm[1], arm[0]])
        sides = []
        for sign in (1.0, -1.0):
            low, high = 0.0, 3.0
            for _step in range(24):
                mid = 0.5 * (low + high)
                point = center_len * arm + sign * mid * perp
                if _point_clear(point, rails, rf_gap):
                    low = mid
                else:
                    high = mid
            sides.append(low)
        apertures.append((arm, sides[0], sides[1]))
    return apertures


def _arm_aperture_radius(direction, center_len, apertures):
    """Radius where a ray meets an arm aperture, or infinity.

    The aperture is the channel cross-section at ``center_len``. A ray aimed
    at a rail misses it and runs on to the rail.
    """
    best = np.inf
    direction = np.asarray(direction, dtype=float)
    for arm, lat_pos, lat_neg in apertures:
        alignment = float(np.dot(direction, arm))
        if alignment <= 1e-8:
            continue
        radius = float(center_len) / alignment
        perp = np.array([-arm[1], arm[0]])
        lateral = radius * float(np.dot(direction, perp))
        if -lat_neg - 1e-9 <= lateral <= lat_pos + 1e-9:
            best = min(best, radius)
    return best


def _inside_polygon(poly, points):
    """Even-odd test for many points. ``points`` has shape ``(n, 2)``."""
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x0 = poly[:, 0][None, :]
    y0 = poly[:, 1][None, :]
    x1 = np.roll(poly[:, 0], -1)[None, :]
    y1 = np.roll(poly[:, 1], -1)[None, :]
    dy = y1 - y0
    crosses = ((y0 > y) != (y1 > y)) & (np.abs(dy) >= 1e-15)
    x_cross = (x1 - x0) * (y - y0) / np.where(np.abs(dy) < 1e-15, 1.0, dy) + x0
    hits = crosses & (x < x_cross)
    return np.sum(hits, axis=1) % 2 == 1


def _rail_distance(points, rail):
    """Distance from each point to a closed rail, and whether it is inside."""
    start = rail
    stop = np.roll(rail, -1, axis=0)
    edge = stop - start
    length2 = np.sum(edge * edge, axis=1)
    relative = points[:, None, :] - start[None, :, :]
    fraction = np.sum(relative * edge[None, :, :], axis=2) / np.maximum(length2, 1e-30)
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[:, :, None] * edge[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)
    return distance, _inside_polygon(rail, points)


def _center_polygon(rails, rf_gap, center_len, n_angles=360):
    """Center electrode for this RF design.

    Toward each rail the boundary stops one RF gap from that rail, so a new
    spline does not open a hole. Down each arm it stops on the channel
    cross-section at ``center_len``, and the segmented inner electrodes begin
    after the DC gap.
    """
    apertures = _channel_apertures(rails, center_len, rf_gap)
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    caps = np.empty(len(angles))
    for index, direction in enumerate(directions):
        aperture = _arm_aperture_radius(direction, center_len, apertures)
        rail_hit = min(
            _ray_radius(np.vstack([rail, rail[:1]]), direction) for rail in rails
        )
        cap = min(aperture, rail_hit)
        if not np.isfinite(cap):
            raise ValueError("center DC ray does not meet a rail or an arm aperture")
        caps[index] = cap
    low = np.zeros(len(angles))
    high = caps.copy()
    for _step in range(24):
        mid = 0.5 * (low + high)
        points = directions * mid[:, None]
        clear = np.ones(len(angles), dtype=bool)
        for rail in rails:
            distance, inside = _rail_distance(points, rail)
            clear &= ~inside
            clear &= distance >= rf_gap
        low = np.where(clear, mid, low)
        high = np.where(clear, high, mid)
    polygon = _ccw(directions * low[:, None])
    if len(polygon) < 3 or _signed_area(polygon) <= 1e-8:
        raise ValueError("center DC electrode has no area")
    return polygon


def _crossing_y(curve, x, sign):
    """Y where ``curve`` crosses a vertical line, on the ``sign`` side of y = 0."""
    found = []
    x = float(x)
    for start, stop in zip(curve[:-1], curve[1:]):
        dx0 = start[0] - x
        dx1 = stop[0] - x
        if dx0 * dx1 > 0.0:
            continue
        if start[0] == stop[0]:
            values = (start[1], stop[1])
        else:
            t = (x - start[0]) / (stop[0] - start[0])
            if t < -1e-9 or t > 1.0 + 1e-9:
                continue
            values = (start[1] + t * (stop[1] - start[1]),)
        for y in values:
            if y * sign > 0.0:
                found.append(float(y))
    if not found:
        return np.nan
    return min(found, key=abs)


def _inner_spans(cutoff, center_len, n_inner, dc_gap):
    """Axial intervals on the negative-x arm, nearest the center first.

    Each item is ``(x_far, x_near)`` with ``x_far < x_near < 0``.
    """
    width = (cutoff - center_len - n_inner * dc_gap) / n_inner
    if width <= 0.0:
        raise ValueError(
            "cutoff {:.4g} is too short for {} inner electrodes".format(cutoff, n_inner)
        )
    near_edge = -(center_len + dc_gap)
    spans = []
    for _index in range(n_inner):
        far_edge = near_edge - width
        spans.append((far_edge, near_edge))
        near_edge = far_edge - dc_gap
    return spans


def _arm_polygon(x_far, x_near, upper, lower, n_samples=24):
    """One inner electrode. Its long edges are the RF-gap offset of the rails."""
    if x_near - x_far < 1e-6:
        raise ValueError("inner electrode has no length")
    xs = np.linspace(x_far, x_near, n_samples)
    top = []
    bottom = []
    for x in xs:
        y_upper = _crossing_y(upper, x, +1.0)
        y_lower = _crossing_y(lower, x, -1.0)
        if not np.isfinite(y_upper) or not np.isfinite(y_lower):
            raise ValueError("inner electrode leaves the RF channel")
        if y_upper - y_lower < 1e-3:
            break
        top.append([x, y_upper])
        bottom.append([x, y_lower])
    if len(top) < 2:
        raise ValueError("inner electrode is covered by the RF gap")
    return _ccw(np.vstack([top, bottom[::-1]]))


def _clip_halfplane(poly, limit, keep_above):
    """Sutherland–Hodgman clip against a horizontal line."""
    poly = np.asarray(poly, dtype=float)

    def inside(point):
        if keep_above:
            return point[1] >= limit
        return point[1] <= limit

    def crossing(start, stop):
        dy = stop[1] - start[1]
        if abs(dy) < 1e-15:
            t = 0.0
        else:
            t = (limit - start[1]) / dy
        t = min(max(float(t), 0.0), 1.0)
        return start + t * (stop - start)

    output = []
    for index in range(len(poly)):
        previous = poly[index - 1]
        current = poly[index]
        previous_in = inside(previous)
        current_in = inside(current)
        if current_in:
            if not previous_in:
                output.append(crossing(previous, current))
            output.append(current)
        elif previous_in:
            output.append(crossing(previous, current))
    polygon = _ccw(np.vstack(output))
    if len(polygon) < 3 or _signed_area(polygon) <= 1e-8:
        raise ValueError("split removed a tilt or outer electrode")
    return polygon


def _sector_spline(arm_end, radial, dc_gap, sample_size):
    """Outer boundary of a tilt or outer band, symmetric about +x."""
    tip = (radial - dc_gap / 2.0) / np.sqrt(3.0) * 2.0
    top = _arm_offset_point(arm_end, radial)
    middle = 0.5 * (top + np.array([tip, 0.0]))
    points = np.vstack([
        top,
        middle,
        [tip, 0.0],
        middle * np.array([1.0, -1.0]),
        top * np.array([1.0, -1.0]),
    ])
    return _spline_curve(points, sample_size)


def _split_band(outer_curve, inner_curve, dc_gap):
    """Upper and lower electrodes of one band. Both curves run top to bottom."""
    band = _ccw(np.vstack([outer_curve, inner_curve[::-1]]))
    upper = _clip_halfplane(band, dc_gap / 2.0, keep_above=True)
    lower = _clip_halfplane(band, -dc_gap / 2.0, keep_above=False)
    return upper, lower


def _as_electrode(name, path):
    polygon = _ccw(path)
    if len(polygon) < 3 or _signed_area(polygon) <= 1e-8:
        raise ValueError("electrode {} has no area".format(name))
    return PolygonPixelElectrode(name=name, paths=[np.asarray(polygon, dtype=float)])


def _segmented_dc_electrodes(
    inner_spline,
    outer_spline,
    rail,
    rf_sep,
    rf_w,
    sample_size,
    n_inner,
    cutoff,
    rf_gap,
    dc_gap,
    outer_tilt,
    outer_length,
):
    """Addressable DC electrodes for the three arms and the three RF rails."""
    if n_inner < 1:
        raise ValueError("a junction needs at least one inner electrode per arm")
    if min(cutoff, rf_gap, dc_gap, outer_tilt, outer_length) <= 0.0:
        raise ValueError("DC lengths and gaps must be positive")
    if rf_gap >= rf_sep / 2.0:
        raise ValueError("RF gap closes the channel")
    center_len = 0.5 * ION_HEIGHT
    # Along-arm distance of the outer spline's end. The straight arm then
    # continues out to the cutoff.
    arm_hat = np.array([np.cos(_ARM_ANGLE), np.sin(_ARM_ANGLE)])
    arm_end = max(float(cutoff), float(np.dot(outer_spline[0], arm_hat)))
    rail = np.asarray(rail, dtype=float)
    rails = [_rotate(rail, turn * 2.0 * np.pi / 3.0) for turn in range(3)]
    inner_edge = _extend_curve(inner_spline, arm_end, rf_sep / 2.0)
    outer_edge = _extend_curve(outer_spline, arm_end, rf_sep / 2.0 + rf_w)
    inner_edges = [_rotate(inner_edge, turn * 2.0 * np.pi / 3.0) for turn in range(3)]
    outer_edges = [_rotate(outer_edge, turn * 2.0 * np.pi / 3.0) for turn in range(3)]

    # Arm 0 is negative x. Rail 1 (120 degrees) is its upper wall and rail 2
    # its lower wall. The walls used here are already inset by the RF gap.
    upper_wall = _offset_away(inner_edges[1], rf_gap, outer_edges[1])
    lower_wall = _offset_away(inner_edges[2], rf_gap, outer_edges[2])
    center = _center_polygon(rails, rf_gap, center_len)
    arm_paths = []
    for x_far, x_near in _inner_spans(cutoff, center_len, n_inner, dc_gap):
        arm_paths.append(_arm_polygon(x_far, x_near, upper_wall, lower_wall))

    tilt_radial = rf_sep / 2.0 + rf_w + rf_gap + outer_tilt
    outer_radial = tilt_radial + dc_gap + outer_length
    tilt_outer = _sector_spline(arm_end, tilt_radial, dc_gap, sample_size)
    outer_outer = _sector_spline(arm_end, outer_radial, dc_gap, sample_size)
    tilt_inner = _offset_away(outer_edge, rf_gap, inner_edge)
    # One DC gap outside the tilt band.
    outer_inner = _offset_away(tilt_outer, dc_gap, tilt_inner)
    tilt_upper, tilt_lower = _split_band(tilt_outer, tilt_inner, dc_gap)
    outer_upper, outer_lower = _split_band(outer_outer, outer_inner, dc_gap)

    electrodes = [_as_electrode("dc_center", center)]
    for arm in range(3):
        rotation = arm * 2.0 * np.pi / 3.0
        for index, path in enumerate(arm_paths):
            electrodes.append(_as_electrode(
                "dc_inner_{}_{}".format(arm, index),
                _rotate(path, rotation),
            ))
    for rail in range(3):
        rotation = rail * 2.0 * np.pi / 3.0
        electrodes.append(_as_electrode(
            "dc_tilt_{}_0".format(rail), _rotate(tilt_upper, rotation)
        ))
        electrodes.append(_as_electrode(
            "dc_tilt_{}_1".format(rail), _rotate(tilt_lower, rotation)
        ))
    for rail in range(3):
        rotation = rail * 2.0 * np.pi / 3.0
        electrodes.append(_as_electrode(
            "dc_outer_{}_0".format(rail), _rotate(outer_upper, rotation)
        ))
        electrodes.append(_as_electrode(
            "dc_outer_{}_1".format(rail), _rotate(outer_lower, rotation)
        ))
    return electrodes


def build_junction(
    control_points,
    rf_sep=RF_SEP,
    rf_w=RF_WIDTH,
    sample_size=100,
    n_inner=N_INNER,
    cutoff=CUTOFF_LENGTH,
    rf_gap=RF_GAP,
    dc_gap=DC_GAP,
    outer_tilt=OUTER_TILT,
    outer_length=OUTER_LENGTH,
):
    """Electrode system for one Y junction.

    The RF electrode is named ``r`` and has ``rf = 1``. Every other electrode
    is a DC electrode at zero volts. ``dc_electrode_names`` lists them. The
    ion channels are at 180, 60, and 300 degrees. The RF rails bulge toward
    0, 120, and 240 degrees.
    """
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

    rails = []
    for turn in range(3):
        rails.append(_rotate(rail, turn * 2.0 * np.pi / 3.0))
    dc_electrodes = _segmented_dc_electrodes(
        inner, outer, rail, rf_sep, rf_w, sample_size,
        int(n_inner), float(cutoff), float(rf_gap), float(dc_gap),
        float(outer_tilt), float(outer_length),
    )
    system = System([
        PolygonPixelElectrode(name="r", paths=[np.asarray(path, dtype=float) for path in rails]),
        *dc_electrodes,
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
