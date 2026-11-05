"""Design-vector packing and the sampling distribution."""

import unittest

import numpy as np

from ion_npe.constants import CROSS_CAP_M, HEIGHT_CAP_M, L_SCALE
from ion_npe.design import BoxGaussian, DesignSpace, within_transport_caps


class DesignTests(unittest.TestCase):
    def test_nominal_round_trip_and_names(self):
        space = DesignSpace()
        design = space.nominal_design()
        vector = space.pack(design)
        restored = space.unpack(vector)
        np.testing.assert_allclose(restored.junction, design.junction)
        np.testing.assert_allclose(restored.arm_xyz, design.arm_xyz)
        np.testing.assert_allclose(restored.interior_xyz, design.interior_xyz)
        self.assertEqual(restored.curves, design.curves)
        self.assertEqual(restored.z_arm, design.z_arm)
        self.assertEqual(restored.z_bend, design.z_bend)
        self.assertEqual(restored.z_cross, design.z_cross)
        self.assertEqual(restored.cross_s, design.cross_s)
        self.assertAlmostEqual(vector[space.names.index("c0_x")], -0.7263)
        self.assertEqual(vector[space.names.index("bend_z")], 1.0)
        self.assertEqual(vector[space.names.index("cross_z")], 1.0)
        self.assertEqual(vector[space.names.index("c6_z")], 1.0)
        self.assertEqual(vector[space.names.index("curve_a")], 3.0)
        self.assertEqual(vector[space.names.index("curve_b")], 1.5)
        self.assertEqual(vector[space.names.index("cross_s")], 0.1)
        self.assertNotIn("z_arm", space.names)
        self.assertEqual(design.z_arm, 1.0)
        self.assertEqual(design.z_cross, 1.0)
        self.assertEqual(len(design.interior_xyz), 7)
        spacing = np.diff(design.interior_xyz[:, 0])
        self.assertLess(spacing[-1], spacing[0])
        self.assertTrue(np.all(vector >= space.lower))
        self.assertTrue(np.all(vector <= space.upper))

    def test_samples_stay_inside_a_shrunk_box(self):
        space = DesignSpace()
        rng = np.random.default_rng(0)
        drawn = space.sample(rng, 20, fraction=0.4)
        center = space.nominal()
        lo = center + 0.4 * (space.lower - center)
        hi = center + 0.4 * (space.upper - center)
        self.assertTrue(np.all(drawn >= lo - 1e-12))
        self.assertTrue(np.all(drawn <= hi + 1e-12))

    def test_height_and_crossing_are_capped_at_ten_micrometers(self):
        space = DesignSpace()
        for name in space.names:
            if not name.endswith("_z"):
                continue
            index = space.names.index(name)
            self.assertAlmostEqual((space.upper[index] - 1.0) * L_SCALE, HEIGHT_CAP_M)
            self.assertAlmostEqual((1.0 - space.lower[index]) * L_SCALE, HEIGHT_CAP_M)
        cross = space.names.index("cross_s")
        self.assertAlmostEqual(space.upper[cross] * L_SCALE, CROSS_CAP_M)
        drawn = space.sample(np.random.default_rng(2), 24, fraction=0.45, cap_fraction=1.0)
        for row in drawn:
            design = space.unpack(row)
            self.assertTrue(within_transport_caps(design))
            self.assertLessEqual(design.cross_s * L_SCALE, CROSS_CAP_M + 1e-12)
        outside = space.nominal_design()
        outside.cross_s = CROSS_CAP_M / L_SCALE + 0.05
        self.assertFalse(within_transport_caps(outside))

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
