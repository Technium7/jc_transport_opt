"""Trajectory geometry, including a free ion height."""

import unittest

import numpy as np

from ion_npe.design import DesignSpace, N_INTERIOR, symmetric_gate_offsets
from ion_npe.trajectory import (
    MIRROR_HAT,
    control_polygon,
    end_point,
    height_within_cap,
    path_geometry,
    s_of_tau,
    time_gate_samples,
)


class TrajectoryTests(unittest.TestCase):
    def test_time_law_starts_and_ends_at_rest(self):
        tau = np.linspace(0.0, 1.0, 40)
        distance = s_of_tau(tau, (3.0, 1.5), n_gates=9)
        self.assertAlmostEqual(distance[0], 0.0, places=8)
        self.assertAlmostEqual(distance[-1], 1.0, places=8)
        self.assertTrue(np.all(np.diff(distance) >= -1e-8))
        zeros = np.zeros(8)
        np.testing.assert_allclose(
            s_of_tau(tau, (3.0, 1.5), n_gates=9, gate_offsets=zeros),
            distance,
        )
        shifted = zeros.copy()
        shifted[0] = 0.04
        _knots, samples = time_gate_samples((3.0, 1.5), 9, shifted)
        _knots, nominal_samples = time_gate_samples((3.0, 1.5), 9)
        self.assertGreater(samples[1], nominal_samples[1])
        self.assertAlmostEqual(samples[0], 0.0, places=8)
        self.assertAlmostEqual(samples[-1], 1.0, places=8)

    def test_a_design_waveform_is_symmetric_about_mid_path(self):
        design = DesignSpace().nominal_design()
        design.gate_offsets = np.array([0.04, -0.02, 0.03, 0.01, 0.08, -0.05, 0.02, -0.01])
        gates = design.gate_offsets
        self.assertEqual(len(gates), 8)
        self.assertAlmostEqual(float(gates[0]), -float(gates[7]))
        self.assertAlmostEqual(float(gates[1]), -float(gates[6]))
        self.assertAlmostEqual(float(gates[2]), -float(gates[5]))
        self.assertAlmostEqual(float(gates[3]), -float(gates[4]))
        np.testing.assert_allclose(symmetric_gate_offsets(gates), gates)
        tau = np.linspace(0.0, 1.0, 241)
        distance = s_of_tau(tau, design.curves, design.n_time_gates, gates)
        np.testing.assert_allclose(distance + distance[::-1], 1.0, atol=1e-8)

    def test_a_large_gate_shift_is_rejected(self):
        from ion_npe.schedule import schedule_is_admissible

        self.assertTrue(schedule_is_admissible(np.zeros(8), (3.0, 1.5), 9))
        too_far = np.zeros(8)
        too_far[0] = 0.2
        too_far[-1] = -0.2
        self.assertFalse(schedule_is_admissible(too_far, (3.0, 1.5), 9))

    def test_constant_height_path(self):
        design = DesignSpace().nominal_design()
        path = path_geometry(design, n=601)
        np.testing.assert_allclose(path[:, 2], 1.0, atol=1e-8)
        np.testing.assert_allclose(path[0, :2], [design.x_start, 0.0], atol=1e-8)
        np.testing.assert_allclose(path[-1, :2], end_point(design.x_start), atol=1e-6)
        mid = path[len(path) // 2]
        np.testing.assert_allclose(mid[:2], design.cross_s * MIRROR_HAT, atol=0.03)
        np.testing.assert_allclose(mid[2], design.z_cross, atol=1e-6)

    def test_height_follows_the_bend_controls(self):
        design = DesignSpace().nominal_design()
        design.interior_z = np.full(N_INTERIOR, 0.85)
        polygon = control_polygon(design)
        self.assertAlmostEqual(polygon[0, 2], 1.0)
        self.assertAlmostEqual(polygon[-1, 2], 1.0)
        self.assertEqual(len(polygon), 2 + N_INTERIOR + 2)
        path = path_geometry(design, n=601)
        self.assertAlmostEqual(path[0, 2], 1.0, places=6)
        self.assertAlmostEqual(path[-1, 2], 1.0, places=6)
        self.assertLess(path[:, 2].min(), 0.95)
        self.assertAlmostEqual(path[len(path) // 2, 2], 1.0, places=2)

    def test_bend_radius_scales_the_developed_bow(self):
        design = DesignSpace().nominal_design()
        nominal_xy = design.interior_xyz[:, :2].copy()
        design.bend_r *= 2.0
        doubled = design.interior_xyz[:, :2].copy()
        design.bend_r = 0.0
        spine = design.interior_xyz[:, :2]
        np.testing.assert_allclose(spine + 2.0 * (nominal_xy - spine), doubled)
        path = path_geometry(design, n=601)
        self.assertAlmostEqual(path[0, 2], 1.0, places=6)
        self.assertAlmostEqual(path[-1, 2], 1.0, places=6)

    def test_spacing_keeps_closing_through_the_junction(self):
        design = DesignSpace().nominal_design()
        polygon = control_polygon(design)
        tail = polygon[2:, :2]
        chords = np.linalg.norm(np.diff(tail, axis=0), axis=1)
        self.assertTrue(np.all(np.diff(chords) < 0.0))
        self.assertAlmostEqual(chords[-1], design.end_handle, places=6)
        self.assertLess(chords[-1], chords[-2])

    def test_nominal_height_is_inside_the_cap_and_a_tall_bend_is_not(self):
        design = DesignSpace().nominal_design()
        self.assertTrue(height_within_cap(design))
        design.interior_z = np.full(N_INTERIOR, 1.8)
        self.assertFalse(height_within_cap(design))


if __name__ == "__main__":
    unittest.main()
