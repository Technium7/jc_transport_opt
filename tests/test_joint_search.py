"""Selection helpers for the joint search. No ion dynamics."""

import unittest

import numpy as np

from ion_npe.design import DesignSpace, expand_free_gates
from ion_npe.joint import (
    N_FREE_GATES, N_GATES, _bounds, _draw_gaussian, _gaussian_std, _latin_bounds,
    _pick_lowest, _unpack, project_joint_row,
)


class JointSearchTests(unittest.TestCase):
    def test_pick_lowest_is_best_first(self):
        scores = np.array([4.0, 0.5, 3.0, 0.2])
        self.assertEqual(_pick_lowest(scores, 2), [3, 1])

    def test_search_stores_four_outbound_gates(self):
        space = DesignSpace()
        self.assertEqual(N_GATES, 9)
        self.assertEqual(N_FREE_GATES, 4)
        lower, upper = _bounds(space)
        self.assertEqual(len(lower), space.dim + N_FREE_GATES)
        free = np.array([0.04, -0.01, 0.02, 0.01])
        design = _unpack(space, np.concatenate([space.nominal(), free]))
        out = design.gate_offsets
        np.testing.assert_allclose(out, expand_free_gates(free))
        self.assertEqual(float(out[4]), 0.0)
        self.assertAlmostEqual(float(out[0]), 0.04)
        self.assertAlmostEqual(float(out[1]), -0.01)
        self.assertAlmostEqual(float(out[2]), 0.02)
        self.assertAlmostEqual(float(out[3]), 0.01)
        self.assertAlmostEqual(float(out[5]), -0.01)

    def test_a_saved_nine_gate_row_drops_the_mirror(self):
        space = DesignSpace()
        legacy = np.array([0.04, -0.01, 0.02, 0.01, 0.9, -0.01, -0.02, 0.01, -0.04])
        row = project_joint_row(np.concatenate([space.nominal(), legacy]))
        self.assertEqual(len(row), space.dim + N_FREE_GATES)
        np.testing.assert_allclose(row[-N_FREE_GATES:], [0.04, -0.01, 0.02, 0.01])
        design = _unpack(space, row)
        self.assertEqual(float(design.gate_offsets[4]), 0.0)

    def test_latin_box_is_a_neighborhood_of_the_nominal_path(self):
        space = DesignSpace()
        lower, upper = _bounds(space)
        nominal = np.concatenate([space.nominal(), np.zeros(N_FREE_GATES)])
        lo, hi = _latin_bounds(space, lower, upper, center=nominal, scale=1.0)
        self.assertTrue(np.all(lo >= lower - 1e-12))
        self.assertTrue(np.all(hi <= upper + 1e-12))
        self.assertTrue(np.all(lo < hi))
        self.assertTrue(np.all(lo <= nominal))
        self.assertTrue(np.all(nominal <= hi))
        # Junction span is 20% of the full box; the first coordinate is ai.
        self.assertAlmostEqual(float(hi[0] - lo[0]), 0.40 * float(upper[0] - lower[0]))
        # Free gates stay within 0.08 of the path.
        self.assertAlmostEqual(float(hi[-1] - lo[-1]), 0.16)
        # A larger scale still cannot leave the legal box.
        wide_lo, wide_hi = _latin_bounds(space, lower, upper, center=nominal, scale=10.0)
        np.testing.assert_allclose(wide_lo, lower)
        np.testing.assert_allclose(wide_hi, upper)

    def test_gaussian_variance_is_the_uniform_range(self):
        space = DesignSpace()
        lower, upper = _bounds(space)
        std = _gaussian_std(lower, upper)
        np.testing.assert_allclose(std ** 2, (upper - lower) ** 2 / 12.0)
        mean = np.concatenate([space.nominal(), np.zeros(N_FREE_GATES)])
        rows = _draw_gaussian(np.random.default_rng(0), 30, mean, lower, upper, space)
        self.assertEqual(rows.shape, (30, len(lower)))
        self.assertTrue(np.all(rows >= lower - 1e-12))
        self.assertTrue(np.all(rows <= upper + 1e-12))
        self.assertTrue(np.all(np.isfinite(rows)))


if __name__ == "__main__":
    unittest.main()
