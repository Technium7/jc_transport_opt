"""Design-vector packing and the sampling distribution."""

import unittest

import numpy as np

from ion_npe.constants import CROSS_CAP_M, HEIGHT_CAP_M, L_SCALE
from ion_npe.design import BoxGaussian, DesignSpace, N_INTERIOR, within_transport_caps


class DesignTests(unittest.TestCase):
    def test_nominal_round_trip_and_names(self):
        space = DesignSpace()
        design = space.nominal_design()
        vector = space.pack(design)
        restored = space.unpack(vector)
        np.testing.assert_allclose(restored.junction, design.junction)
        self.assertEqual(restored.cross_s, design.cross_s)
        np.testing.assert_allclose(restored.interior_z, design.interior_z)
        np.testing.assert_allclose(restored.interior_xyz, design.interior_xyz)
        self.assertEqual(restored.curves, design.curves)
        self.assertEqual(restored.z_arm, design.z_arm)
        self.assertEqual(restored.z_bend, design.z_bend)
        self.assertEqual(restored.z_cross, design.z_cross)
        self.assertAlmostEqual(vector[space.names.index("cross_s")] * L_SCALE * 1e6, 2.0)
        from ion_npe.design import interior_xy_from_s
        developed = interior_xy_from_s(0.1)
        self.assertAlmostEqual(developed[0, 0], -0.7263, places=3)
        self.assertAlmostEqual(developed[0, 1], 0.0009, places=3)
        self.assertIn("h0_z", space.names)
        self.assertIn("h6_z", space.names)
        self.assertNotIn("c0_z", space.names)
        self.assertTrue(np.allclose(restored.height_z, 1.0))
        z_index = space.names.index("h0_z")
        self.assertAlmostEqual(space.lower[z_index] * L_SCALE, 50e-6 - HEIGHT_CAP_M)
        self.assertAlmostEqual(space.upper[z_index] * L_SCALE, 50e-6 + HEIGHT_CAP_M)
        self.assertNotIn("curve_a", space.names)
        self.assertNotIn("curve_b", space.names)
        self.assertEqual(design.curves, (3.0, 1.5))
        self.assertEqual(restored.curves, (3.0, 1.5))
        self.assertNotIn("bend_r", space.names)
        self.assertNotIn("c0_x", space.names)
        self.assertNotIn("c0_y", space.names)
        self.assertNotIn("c0_r", space.names)
        self.assertNotIn("a0_x", space.names)
        self.assertNotIn("z_arm", space.names)
        self.assertNotIn("a0_z", space.names)
        self.assertNotIn("z_cross", space.names)
        moved = vector.copy()
        moved[space.names.index("h0_z")] = 1.2
        moved[space.names.index("h6_z")] = 0.8
        raised = space.unpack(moved)
        self.assertEqual(raised.z_arm, 1.0)
        self.assertAlmostEqual(raised.height_z[0], 1.2)
        self.assertAlmostEqual(raised.height_z[-1], 0.8)
        self.assertAlmostEqual(raised.z_cross, 0.8)
        from ion_npe.trajectory import path_geometry
        path = path_geometry(raised, n=601)
        self.assertAlmostEqual(path[0, 2], 1.0)
        self.assertAlmostEqual(path[-1, 2], 1.0)
        self.assertAlmostEqual(path[len(path) // 2, 2], 0.8, places=5)
        np.testing.assert_allclose(path[:, 2], path[::-1, 2], atol=1e-8)
        self.assertEqual(len(design.interior_z), 7)
        self.assertEqual(N_INTERIOR, 7)
        spacing = np.diff(design.interior_xyz[:, 0])
        self.assertLess(spacing[-1], spacing[0])
        self.assertTrue(np.all(vector >= space.lower))
        self.assertTrue(np.all(vector <= space.upper))
        self.assertEqual(design.n_time_gates, 10)
        self.assertEqual(len(design.gate_offsets), 9)

    def test_samples_stay_inside_a_shrunk_box(self):
        space = DesignSpace()
        rng = np.random.default_rng(0)
        drawn = space.sample(rng, 20, fraction=0.4)
        center = space.nominal()
        lo = center + 0.4 * (space.lower - center)
        hi = center + 0.4 * (space.upper - center)
        self.assertTrue(np.all(drawn >= lo - 1e-12))
        self.assertTrue(np.all(drawn <= hi + 1e-12))

    def test_radius_and_height_stay_inside_the_cap(self):
        space = DesignSpace()
        self.assertTrue(any(name.endswith("_z") for name in space.names))
        index = space.names.index("cross_s")
        self.assertAlmostEqual(space.upper[index] * L_SCALE, CROSS_CAP_M)
        self.assertEqual(space.lower[index], 0.0)
        drawn = space.sample(np.random.default_rng(2), 24, fraction=0.45, cap_fraction=1.0)
        heights = []
        for row in drawn:
            design = space.unpack(row)
            self.assertTrue(within_transport_caps(design))
            self.assertLessEqual(design.cross_s * L_SCALE, CROSS_CAP_M + 1e-12)
            self.assertLessEqual(np.max(np.abs(design.height_z - 1.0)) * L_SCALE, HEIGHT_CAP_M + 1e-12)
            heights.append(design.height_z.copy())
        self.assertGreater(np.std(np.vstack(heights)), 1e-3)
        outside = space.nominal_design()
        outside.cross_s = CROSS_CAP_M / L_SCALE + 0.05
        self.assertFalse(within_transport_caps(outside))
        high = space.nominal_design()
        high.height_z = np.full(7, 1.0 + HEIGHT_CAP_M / L_SCALE + 0.05)
        self.assertFalse(within_transport_caps(high))

    def test_latin_puts_one_sample_in_each_bin(self):
        space = DesignSpace()
        n = 16
        drawn = space.sample_latin(np.random.default_rng(1), n, fraction=0.55, cap_fraction=1.0)
        lo, hi = space._bounds(0.55, 1.0)
        unit = (drawn - lo) / (hi - lo)
        strata = np.minimum(np.floor(unit * n).astype(int), n - 1)
        for column in range(space.dim):
            self.assertEqual(len(set(strata[:, column])), n)

    def test_gaussian_moves_onto_identical_elites(self):
        space = DesignSpace()
        dist = BoxGaussian.around(space, std=0.2)
        elite = np.repeat(space.nominal()[None, :], 5, axis=0)
        dist.update(elite, smooth=1.0, std_floor=0.02, std_cap=0.5)
        np.testing.assert_allclose(dist.unit_mean, np.clip(space.to_unit(space.nominal()), 0, 1))
        np.testing.assert_allclose(dist.unit_std, np.full(space.dim, 0.02))
        drawn = dist.sample(np.random.default_rng(1), 30)
        self.assertTrue(np.all(drawn >= space.lower - 1e-12))
        self.assertTrue(np.all(drawn <= space.upper + 1e-12))


if __name__ == "__main__":
    unittest.main()
