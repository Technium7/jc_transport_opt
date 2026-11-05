"""Selection helpers for the joint search. No ion dynamics."""

import unittest

import numpy as np

from ion_npe.design import DesignSpace
from ion_npe.joint import _pick_lowest, _unpack


class JointSearchTests(unittest.TestCase):
    def test_pick_lowest_is_best_first(self):
        scores = np.array([4.0, 0.5, 3.0, 0.2])
        self.assertEqual(_pick_lowest(scores, 2), [3, 1])

    def test_a_joint_design_stores_a_symmetric_waveform(self):
        space = DesignSpace()
        row = np.concatenate([space.nominal(), [0.04, -0.01, 0.02, 0.03, 0.05]])
        design = _unpack(space, row)
        gates = design.gate_offsets
        self.assertEqual(float(gates[2]), 0.0)
        self.assertAlmostEqual(float(gates[0]), -float(gates[4]))
        self.assertAlmostEqual(float(gates[1]), -float(gates[3]))


if __name__ == "__main__":
    unittest.main()
