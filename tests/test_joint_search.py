"""Selection helpers for the joint search. No ion dynamics."""

import unittest

import numpy as np

from ion_npe.design import DesignSpace
from ion_npe.joint import N_GATES, _pick_lowest, _unpack


class JointSearchTests(unittest.TestCase):
    def test_pick_lowest_is_best_first(self):
        scores = np.array([4.0, 0.5, 3.0, 0.2])
        self.assertEqual(_pick_lowest(scores, 2), [3, 1])

    def test_a_joint_design_stores_a_symmetric_waveform(self):
        space = DesignSpace()
        self.assertEqual(N_GATES, 7)
        gates = np.array([0.04, -0.01, 0.02, 0.03, -0.02, 0.01, -0.04])
        row = np.concatenate([space.nominal(), gates])
        design = _unpack(space, row)
        out = design.gate_offsets
        self.assertEqual(len(out), 7)
        self.assertAlmostEqual(float(out[0]), -float(out[6]))
        self.assertAlmostEqual(float(out[1]), -float(out[5]))
        self.assertAlmostEqual(float(out[2]), -float(out[4]))
        self.assertEqual(float(out[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
