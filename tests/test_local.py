"""Neighborhood geometry for the local joint search. No ion dynamics."""

import unittest

import numpy as np

from ion_npe.local import blocks, embed, neighborhood_half, box_around
from ion_npe.design import DesignSpace
from ion_npe.joint import _bounds


class LocalNeighborhoodTests(unittest.TestCase):
    def test_waveform_slice_keeps_the_nominal_junction_and_path(self):
        space = DesignSpace()
        lower, upper = _bounds(space)
        center = np.concatenate([space.nominal(), np.zeros(5)])
        group = blocks(len(center))
        moved = embed(center, center[group["waveform"]] + 0.02, group["waveform"])
        self.assertTrue(np.allclose(moved[group["junction"]], center[group["junction"]]))
        self.assertTrue(np.allclose(moved[group["trajectory"]], center[group["trajectory"]]))
        self.assertTrue(np.allclose(moved[group["waveform"]], 0.02))

    def test_height_window_can_leave_50_um_and_gates_cover_a_waveform_shift(self):
        space = DesignSpace()
        lower, upper = _bounds(space)
        half = neighborhood_half(space, lower, upper)
        self.assertGreater(half[-1], 0.07)
        z = [i for i, name in enumerate(space.names) if name.endswith("_z")]
        self.assertTrue(np.all(half[z] > 0.1))
        lo, hi = box_around(np.concatenate([space.nominal(), np.zeros(5)]), half, lower, upper)
        self.assertTrue(np.all(hi > lo))


if __name__ == "__main__":
    unittest.main()
