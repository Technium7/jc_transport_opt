"""Task split for a multi-task SLURM step. No ion dynamics."""

import multiprocessing
import os
import tempfile
import time
import unittest

import numpy as np

import ion_npe.parallel as parallel
from ion_npe.parallel import active_workers, map_joint, merge_parts, serve, stop, task_indices


class _Result:
    def __init__(self, row):
        self.ok = True
        self.quanta = np.array([[float(row[0]), 0.0, 0.0]])


def _fake_joint(row, _kwargs):
    return _Result(row)


class ParallelSplitTests(unittest.TestCase):
    def test_shares_cover_every_row_once(self):
        seen = []
        for rank in range(12):
            seen.extend(task_indices(25, rank, 12))
        self.assertEqual(sorted(seen), list(range(25)))

    def test_merge_puts_successes_back_in_order(self):
        first = np.array([[[1.0, 2.0, 3.0]]])
        second = np.array([[[4.0, 5.0, 6.0]]])
        merged = merge_parts(
            [
                (np.array([0]), np.array([True]), first),
                (np.array([1]), np.array([False]), np.zeros((1, 1, 3))),
                (np.array([2]), np.array([True]), second),
            ],
            3,
        )
        self.assertTrue(np.allclose(merged[0], first[0]))
        self.assertIsNone(merged[1])
        self.assertTrue(np.allclose(merged[2], second[0]))

    def test_task_count_replaces_the_local_pool(self):
        previous = os.environ.get("SLURM_NTASKS")
        os.environ["SLURM_NTASKS"] = "12"
        try:
            self.assertEqual(active_workers(1), 12)
        finally:
            if previous is None:
                os.environ.pop("SLURM_NTASKS", None)
            else:
                os.environ["SLURM_NTASKS"] = previous
        self.assertEqual(active_workers(3), 3)

    def test_two_tasks_collect_disjoint_rows(self):
        keys = ("SLURM_PROCID", "SLURM_NTASKS", "ION_NPE_FARM")
        previous = {key: os.environ.get(key) for key in keys}
        directory = tempfile.mkdtemp()
        os.environ["ION_NPE_FARM"] = directory
        os.environ["SLURM_NTASKS"] = "2"
        os.environ["SLURM_PROCID"] = "0"
        original = parallel._simulate_joint_row
        parallel._simulate_joint_row = _fake_joint

        def worker():
            os.environ["SLURM_PROCID"] = "1"
            os.environ["SLURM_NTASKS"] = "2"
            os.environ["ION_NPE_FARM"] = directory
            serve()

        proc = multiprocessing.get_context("fork").Process(target=worker)
        proc.start()
        try:
            time.sleep(0.3)
            rows = [np.array([float(index), 0.0]) for index in range(4)]
            results = map_joint(rows, {"seeds": (0,)})
            got = [float(result.quanta[0, 0]) for result in results]
            self.assertEqual(got, [0.0, 1.0, 2.0, 3.0])
        finally:
            stop()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
            parallel._simulate_joint_row = original
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
