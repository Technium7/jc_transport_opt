"""Ranking for active collection near the nominal path."""

import unittest

import numpy as np

from ion_npe.active import active_ranking


class ActiveTests(unittest.TestCase):
    def test_low_phonon_uncertain_points_come_first(self):
        mean = np.array([1.0e6, 20.0, 20.0, 5.0e5])
        std = np.array([1.0e5, 1.0, 8.0, 10.0])
        order = active_ranking(mean, std, phonon_quantile=0.5)
        self.assertEqual(set(order.tolist()), {1, 2})
        self.assertEqual(int(order[0]), 2)


if __name__ == "__main__":
    unittest.main()
