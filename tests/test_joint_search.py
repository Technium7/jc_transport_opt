"""Selection helpers for the joint search. No ion dynamics."""

import unittest

import numpy as np

from ion_npe.design import DesignSpace, expand_free_gates
from ion_npe.joint import N_FREE_GATES, N_GATES, _bounds, _pick_lowest, _unpack, project_joint_row


class JointSearchTests(unittest.TestCase):
    def test_pick_lowest_is_best_first(self):
        scores = np.array([4.0, 0.5, 3.0, 0.2])
        self.assertEqual(_pick_lowest(scores, 2), [3, 1])

    def test_search_stores_three_outbound_gates(self):
        space = DesignSpace()
        self.assertEqual(N_GATES, 7)
        self.assertEqual(N_FREE_GATES, 3)
        lower, upper = _bounds(space)
        self.assertEqual(len(lower), space.dim + N_FREE_GATES)
        free = np.array([0.04, -0.01, 0.02])
        design = _unpack(space, np.concatenate([space.nominal(), free]))
        out = design.gate_offsets
        np.testing.assert_allclose(out, expand_free_gates(free))
        self.assertEqual(float(out[3]), 0.0)
        self.assertAlmostEqual(float(out[0]), 0.04)
        self.assertAlmostEqual(float(out[1]), -0.01)
        self.assertAlmostEqual(float(out[2]), 0.02)

    def test_a_saved_seven_gate_row_drops_the_mirror(self):
        space = DesignSpace()
        legacy = np.array([0.04, -0.01, 0.02, 0.9, -0.02, 0.01, -0.04])
        row = project_joint_row(np.concatenate([space.nominal(), legacy]))
        self.assertEqual(len(row), space.dim + N_FREE_GATES)
        np.testing.assert_allclose(row[-N_FREE_GATES:], [0.04, -0.01, 0.02])
        design = _unpack(space, row)
        self.assertEqual(float(design.gate_offsets[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
