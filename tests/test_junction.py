"""Segmented DC electrodes on the Y junction."""

import unittest

import numpy as np
from matplotlib.path import Path

from ion_npe.design import DEFAULT_JUNCTION, DesignSpace
from ion_npe.junction import N_INNER, RF_GAP, build_junction, dc_electrode_names


def _distance_to_rails(points, rails):
    """Minimum distance from each point to the boundaries of the RF rails."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    best = np.full(len(points), np.inf)
    for rail in rails:
        closed = np.vstack([rail, rail[:1]])
        start = closed[:-1]
        stop = closed[1:]
        edge = stop - start
        length2 = np.sum(edge * edge, axis=1)
        relative = points[:, None, :] - start[None, :, :]
        fraction = np.clip(
            np.sum(relative * edge[None, :, :], axis=2) / np.maximum(length2, 1e-30),
            0.0,
            1.0,
        )
        closest = start[None, :, :] + fraction[:, :, None] * edge[None, :, :]
        best = np.minimum(best, np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1))
    return best


def _interior(poly, n=7):
    path = Path(poly)
    xs = np.linspace(poly[:, 0].min(), poly[:, 0].max(), n)
    ys = np.linspace(poly[:, 1].min(), poly[:, 1].max(), n)
    grid = np.array([[x, y] for x in xs for y in ys])
    return grid[path.contains_points(grid)]


class JunctionElectrodeTests(unittest.TestCase):
    def test_nominal_segments_match_the_y_layout(self):
        junction = build_junction(DEFAULT_JUNCTION)
        names = dc_electrode_names()
        self.assertEqual(junction.names, ["r", *names])
        self.assertEqual(junction["r"].rf, 1.0)
        self.assertTrue(np.all(junction.dcs == 0.0))
        self.assertEqual(sum(name.startswith("dc_inner") for name in names), 3 * N_INNER)
        self.assertEqual(sum(name.startswith("dc_tilt") for name in names), 6)
        self.assertEqual(sum(name.startswith("dc_outer") for name in names), 6)
        for electrode in junction:
            if electrode.name == "r":
                continue
            polygon = electrode.paths[0]
            self.assertGreater(len(polygon), 2)
            self.assertTrue(np.isfinite(polygon).all())

        # Arm 0 is the inbound leg. x = -3 sits on the third inner electrode.
        self.assertTrue(Path(junction["dc_inner_0_2"].paths[0]).contains_point((-3.0, 0.0)))
        self.assertFalse(Path(junction["dc_inner_0_1"].paths[0]).contains_point((-3.0, 0.0)))
        junction["dc_inner_0_2"].dc = 1.0
        potential = junction.electrical_potential([[-3.0, 0.0, 1.0]], "dc", 0)
        self.assertGreater(float(potential[0, 0]), 0.0)

    def test_dc_metal_stays_out_of_the_rf(self):
        junction = build_junction(DEFAULT_JUNCTION)
        rails = [Path(path) for path in junction["r"].paths]
        for electrode in junction:
            if electrode.name == "r":
                continue
            samples = _interior(electrode.paths[0])
            self.assertGreater(len(samples), 0, electrode.name)
            for rail in rails:
                self.assertFalse(
                    rail.contains_points(samples).any(),
                    electrode.name,
                )

    def test_arms_repeat_every_120_degrees(self):
        junction = build_junction(DEFAULT_JUNCTION)
        turn = np.array([
            [-0.5, -np.sqrt(3.0) / 2.0],
            [np.sqrt(3.0) / 2.0, -0.5],
        ])
        for index in range(N_INNER):
            first = junction["dc_inner_0_{}".format(index)].paths[0]
            second = junction["dc_inner_1_{}".format(index)].paths[0]
            np.testing.assert_allclose(first @ turn.T, second, atol=1e-12)
        tilt = junction["dc_tilt_0_0"].paths[0]
        np.testing.assert_allclose(
            tilt @ turn.T, junction["dc_tilt_1_0"].paths[0], atol=1e-12
        )

    def test_dc_gap_does_not_fold_back_on_itself(self):
        junction = build_junction(DEFAULT_JUNCTION)
        rails = junction["r"].paths
        # The old miter offset left a 172 degree spur here, 3.5 um long.
        spur = np.array([1.8197, 0.0965])
        for electrode in junction:
            if electrode.name == "r":
                continue
            polygon = electrode.paths[0]
            distance = _distance_to_rails(polygon, rails)
            self.assertGreaterEqual(float(distance.min()), RF_GAP - 2e-3, electrode.name)
            self.assertGreater(np.linalg.norm(polygon - spur, axis=1).min(), 0.02, electrode.name)
            for index in range(len(polygon)):
                previous = polygon[index - 1]
                vertex = polygon[index]
                nxt = polygon[(index + 1) % len(polygon)]
                incoming = vertex - previous
                outgoing = nxt - vertex
                length_in = np.linalg.norm(incoming)
                length_out = np.linalg.norm(outgoing)
                if length_in < 1e-9 or length_out < 1e-9:
                    continue
                turn = abs(np.degrees(np.arctan2(
                    incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                    np.dot(incoming, outgoing),
                )))
                self.assertLess(turn, 150.0, "{} at {}".format(electrode.name, vertex))
        tilt = junction["dc_tilt_0_0"].paths[0]
        self.assertLess(float(_distance_to_rails(tilt, rails).min()), RF_GAP + 2e-3)

    def test_center_stays_one_rf_gap_from_a_moved_rail(self):
        space = DesignSpace()
        controls = space.nominal().copy()
        controls[0] = space.upper[0]
        junction = build_junction(controls[:8])
        center = junction["dc_center"].paths[0]
        rails = junction["r"].paths
        distance = _distance_to_rails(center, rails)
        angles = np.arctan2(center[:, 1], center[:, 0])
        facing = np.zeros(len(center), dtype=bool)
        for target in (0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0):
            delta = np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))
            facing |= delta < np.radians(15.0)
        self.assertGreater(int(np.count_nonzero(facing)), 0)
        self.assertGreaterEqual(float(distance.min()), RF_GAP - 2e-3)
        self.assertLess(float(distance[facing].max()), RF_GAP + 0.01)

    def test_design_box_corners_build(self):
        space = DesignSpace()
        for controls in (space.lower[:8], space.upper[:8]):
            junction = build_junction(controls)
            self.assertEqual(junction.names[0], "r")
            self.assertEqual(len(junction.names), 1 + len(dc_electrode_names()))


if __name__ == "__main__":
    unittest.main()
