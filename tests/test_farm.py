"""Slurm farm: one batch, several ranks, results back in submission order."""

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from ion_npe.farm import Farm, owned_indices, width_for


class FarmTests(unittest.TestCase):
    def test_one_process_keeps_a_width_of_one(self):
        self.assertEqual(width_for(100, 1), 1)
        self.assertEqual(width_for(3, 1), 1)

    def test_ten_nodes_take_eighty_or_whatever_is_left(self):
        self.assertEqual(width_for(100, 10), 80)
        self.assertEqual(width_for(10, 10), 10)
        self.assertEqual(width_for(3, 10), 3)

    def test_ranks_split_the_batch(self):
        self.assertEqual(owned_indices(10, 0, 4), [0, 4, 8])
        self.assertEqual(owned_indices(10, 1, 4), [1, 5, 9])
        self.assertEqual(owned_indices(3, 3, 4), [])

    def test_two_ranks_return_rows_in_order(self):
        rows = np.arange(21, dtype=float).reshape(7, 3)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root = Farm(work, 0, 2, poll_s=0.01, timeout_s=15)
            worker = Farm(work, 1, 2, poll_s=0.01, timeout_s=15)
            holder = {}

            def run_root():
                holder["out"] = root.simulate("echo", rows, {"seeds": [0]})
                root.stop()

            threads = [
                threading.Thread(target=worker.serve),
                threading.Thread(target=run_root),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20)
                self.assertFalse(thread.is_alive())
            sums = [float(item["quanta"][0, 0]) for item in holder["out"]]
        self.assertTrue(np.allclose(sums, rows.sum(axis=1)))

    def test_a_single_rank_does_not_need_a_shared_directory(self):
        farm = Farm(Path("missing-farm-dir"), 0, 1)
        out = farm.simulate("echo", np.array([1.0, 2.0, 3.0]), {"seeds": [0]})
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out[0]["quanta"][0, 0]), 6.0)
        farm.stop()


if __name__ == "__main__":
    unittest.main()
