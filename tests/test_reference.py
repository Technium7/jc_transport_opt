"""The junction pseudopotential matches the developed electrodes."""

import sys
import unittest
from pathlib import Path

import numpy as np

OLD = Path("/Users/JackX/Desktop/2024-2025/TrapElectrodes/y_junction/junction_transport_coopt")


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OLD.is_dir():
            raise unittest.SkipTest("developed transport code is not on this machine")
        sys.path.insert(0, str(OLD))
        import junction as old_junction
        cls.old_junction = old_junction

    def test_pseudopotential_matches(self):
        from ion_npe.design import DEFAULT_JUNCTION
        from ion_npe.junction import build_junction, laplacian_at

        old = self.old_junction.build_junction(DEFAULT_JUNCTION)
        new = build_junction(DEFAULT_JUNCTION)
        xs = np.linspace(-4.0, -0.3, 12)
        ys = np.zeros_like(xs)
        zs = np.ones_like(xs)
        np.testing.assert_allclose(
            laplacian_at(new, xs, ys, zs),
            self.old_junction.laplacian_at(old, xs, ys, zs),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_nominal_transport_at_100_ns(self):
        from ion_npe.constants import ION_HEIGHT_M, L_SCALE, SIM_DT
        from ion_npe.design import DesignSpace
        from ion_npe.simulate import simulate_design

        design = DesignSpace().nominal_design()
        result = simulate_design(
            design,
            seeds=(0,),
            noise_std_Vm=0.0,
            with_ideal=True,
        )
        self.assertTrue(result.ok, result.message)
        self.assertLessEqual(abs(result.n_hold * SIM_DT - 100e-9), 0.5 * SIM_DT)
        self.assertTrue(np.all(np.isfinite(result.quanta)))
        self.assertAlmostEqual(design.z_arm * L_SCALE, ION_HEIGHT_M)
        self.assertAlmostEqual(result.z_range[0] * L_SCALE, ION_HEIGHT_M, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
